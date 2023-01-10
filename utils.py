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
        image = image.resize((256,256))
        transform = transforms.ToTensor()
        return transform(image)

    def __len__(self):
        return len(self.files)

    
    
    
    
    
    

    
    
    
    
