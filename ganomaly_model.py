import math
from typing import Tuple, Union

import torch
from torch import Tensor, nn

class ResBlock(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layer1 = nn.Sequential()
        self.layer1.add_module("Conv2d 1,1", nn.Conv2d(input_channels, input_channels*2, kernel_size=1, stride=1, padding=0, bias=False))
        self.layer1.add_module(f"{input_channels*2}-LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential()
        self.layer2.add_module("Conv2d 3,3", nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3, stride=1, padding=1, bias=False))
        self.layer2.add_module(f"{input_channels*2}-LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Conv2d(input_channels*2, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.output_act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.output_act(x+output)
        return output

class Encoder(nn.Module):

    def __init__(self, img_size: Tuple[int, int], latent_vec_size: int, img_channels: int, n_features: int, res_block_num: int):
        """Encoder Network.
        Args:
            input_size (Tuple[int, int]): Size of input image
            latent_vec_size (int): Size of latent vector z
            img_channels (int): Number of input image channels
            n_features (int): Number of features per convolution layer
            
        """
        super().__init__()
        
        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{img_channels}-{n_features}",
            nn.Conv2d(img_channels, n_features, kernel_size=4, stride=2, padding=2, bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))
        
        self.res_blocks = nn.Sequential()
        for i in range(res_block_num):
            self.res_blocks.add_module(
                f"res_block_{i+1}",
                ResBlock(n_features),
            )
        
        # Create pyramid features to reach latent vector
        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*img_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_features
            out_features = n_features * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
            n_features = out_features
            pyramid_dim = pyramid_dim // 2
            
        self.final_conv_layer = nn.Conv2d(
            n_features,
            latent_vec_size,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        )
        
    def forward(self, input_tensor: Tensor):
        """Return latent vectors."""

        output = self.input_layers(input_tensor)
        output = self.res_blocks(output)
        output = self.pyramid_features(output)
        output = self.final_conv_layer(output)

        return output

class Decoder(nn.Module):
    
    def __init__(self, img_size: Tuple[int, int], latent_vec_size: int, img_channels: int, n_features: int, res_block_num: int):
        
        super().__init__()

        self.latent_input = nn.Sequential()

        # Calculate input channel size to recreate inverse pyramid
        exp_factor = math.ceil(math.log(min(img_size) // 2, 2)) - 2
        n_input_features = n_features * (2**exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose2d(
                latent_vec_size,
                n_input_features,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
        
        # Create inverse pyramid
        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = min(*img_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_input_features
            out_features = n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
            n_input_features = out_features
            pyramid_dim = pyramid_dim // 2
            
        self.res_blocks = nn.Sequential()
        for i in range(res_block_num):
            self.res_blocks.add_module(
                f"res_block_{i+1}",
                ResBlock(n_features),
            )
            
        # Final layers
        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{img_channels}-convt",
            nn.ConvTranspose2d(
                n_input_features,
                img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{img_channels}-tanh", nn.Tanh())
        
    def forward(self, input_tensor: Tensor):
        """Return generated image."""
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        output = self.res_blocks(output)
        output = self.final_layers(output)
        return output

class Generator(nn.Module):
    
    def __init__(self, img_size: Tuple[int, int], latent_vec_size: int, img_channels: int, n_features: int, res_block_num: int):
        
        super().__init__()
        self.encoder1 = Encoder(img_size, latent_vec_size, img_channels, n_features, res_block_num)
        self.decoder = Decoder(img_size, latent_vec_size, img_channels, n_features, res_block_num)
        self.encoder2 = Encoder(img_size, latent_vec_size, img_channels, n_features, res_block_num)
        
    def forward(self, input_tensor: Tensor):
        
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o


class Discriminator(nn.Module):
    
    def __init__(self, img_size: Tuple[int, int], img_channels: int, n_features: int, res_block_num: int):
        super().__init__()
        self.encoder = Encoder(img_size, 1, img_channels, n_features, res_block_num)
        #layers = []
        #for block in encoder.children():
        #    if isinstance(block, nn.Sequential):
        #        layers.extend(list(block.children()))
        #    else:
        #        layers.append(block)

        #self.features = nn.Sequential(*layers[:-1])
        #self.classifier = nn.Sequential(layers[-1])
        #self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input_tensor):
        """Return class of object and features."""
        output = self.encoder(input_tensor)
        putput = output.view(-1,1).squeeze(1)
        return output
        #features = self.features(input_tensor)
        #classifier = self.classifier(features)
        #classifier = classifier.view(-1, 1).squeeze(1)
        #return classifier, features

class Ganomaly(nn.Module):
    def __init__(self, img_size: Tuple[int, int], latent_vec_size: int, img_channels: int, n_features: int, res_block_num: int, device):
        super().__init__()
        super().__init__()
        self.generator: Generator = Generator(
            img_size=img_size,
            latent_vec_size=latent_vec_size,
            img_channels=img_channels,
            n_features=n_features,
            res_block_num=res_block_num,
        ).to(device)
        self.discriminator: Discriminator = Discriminator(
            img_size=img_size,
            img_channels=img_channels,
            n_features=n_features,
            res_block_num=res_block_num,
        ).to(device)
        self.weights_init(self.generator)
        self.weights_init(self.discriminator)
    
    @staticmethod
    def weights_init(module: nn.Module):
        """Initialize DCGAN weights.
        Args:
            module (nn.Module): [description]
        """
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
            
    def forward(self, batch: Tensor) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor], Tensor]:
        """Get scores for batch.
        Args:
            batch (Tensor): Images
        Returns:
            Tensor: Regeneration scores.
        """
        padded_batch = pad_nextpow2(batch)
        fake, latent_i, latent_o = self.generator(padded_batch)
        if self.training:
            return padded_batch, fake, latent_i, latent_o
        return torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)  # convert nx1x1 to n
    
        

        
        
        

