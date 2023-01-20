import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


#def pad_image_1024(image):
#    width, height = image.shape[2], image.shape[1]
#    left = (1024-width)//2
#    right = 1024-width-left
#    top = (1024-height)//2
#    bot = 1024-top-bot
    
#    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value = 0)
    
    
#def pad_image_1024_batch(image_batch):
#    '''
#    This method pads a batch of images to 1024*1024. 
#    '''
#    new_batch=[]
#    for image in image_batch:
#        new_batch.append(pad_image_1024(image))
#   
#    return np.array(new_batch)


def load_mvtec_dataset(category):
    path = "mvtec_anomaly_detection/"+category+"/train/good"
    img_files = os.listdir(path)
    train_set = []
    for img_f in img_files:
        image = Image.open(os.path.join(path, img_f))
        image = image.resize((256,256))
        train_set.append(np.array(image))
    
    return np.array(train_set)


class MvtecDataset(torch.utils.data.Dataset):
    def __init__(self, category):
        self.path = "mvtec_anomaly_detection/"+category+"/train/good"
        self.files = os.listdir(self.path)

    def __getitem__(self, idx):
        f = self.files[idx]
        image = Image.open(os.path.join(self.path, f))
        image = image.resize((256,256), resample = Image.BILINEAR)
        transform = transforms.ToTensor()
        return transform(image)

    def __len__(self):
        return len(self.files)

    
class MvtecDataset_test(torch.utils.data.Dataset):
    def __init__(self, category, condition="anomaly"):
        self.path = "mvtec_anomaly_detection/"+category+"/test/"
        subfiles = os.listdir(self.path)
        self.img_files = []
        for f in subfiles:
            if condition == "anomaly":
                if f != "good":
                    imgs = os.listdir(os.path.join(self.path, f))
                    for img in imgs:
                        self.img_files.append(os.path.join(self.path,f,img))
            else:
                if f == "good":
                    imgs = os.listdir(os.path.join(self.path, f))
                    for img in imgs:
                        self.img_files.append(os.path.join(self.path,f,img))

    def __getitem__(self, idx):
        f = self.img_files[idx]
        image = Image.open(f)
        image = image.resize((256,256), resample = Image.BILINEAR)
        transform = transforms.ToTensor()
        return transform(image)

    def __len__(self):
        return len(self.img_files)

def gradient_penalty(discriminator, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (torch.ones_like(epsilon) - epsilon)

    mixed_scores = discriminator(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
    
    
    
    

    
    
    
    
