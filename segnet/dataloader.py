import numpy as np
import cv2
import json 
from torch.utils.data import Dataset

class RibDataset(Dataset):
    def __init__(self, mode="train", transforms=None, json_path=None):
        self.transforms = transforms
        self.mode = mode
        self.json_path = json_path
        with open(self.json_path,'r') as f:
            data = json.load(f)[self.mode]
        f.close()
        self.imgs = data['imgs']
        self.masks = data['masks']

    def __getitem__(self, idx):

        image = cv2.imread(self.imgs[idx])
        image = (image - image.min()) / (image.max() - image.min())

        mask = np.load(self.masks[idx])
        mask[mask>=1] = 1
        
        mask = mask.transpose(1,2,0)
        if self.transforms:
            augs = self.transforms(image=image, mask=mask)
        return augs["image"], augs["mask"] 
    
    def __len__(self):
        return len(self.imgs)