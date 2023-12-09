import os
import cv2
import torch
from torch.utils.data import Dataset
#from torchvision import transforms

def load_img(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, ipc=None):
        self.data_path = data_path
        self.transform = transform
        self.classes = sorted(os.listdir(data_path))
        self.ipc = ipc

        #self.images = [os.path.join(data_path, image) for image in os.listdir(data_path)]
    def __len__(self):
        return sum(len(files) for _, _, files in os.walk(self.data_path))
    
    def __getitem__(self, idx):
        class_folder = os.path.join(self.data_path, self.classes[idx // self.ipc])
        img_name = f"{self.classes[idx // self.ipc]}_{idx % self.ipc}.png"
        img_path = os.path.join(class_folder, img_name)
        img = load_img(img_path)

        if self.transform:
            img = self.transform(img)
        return img, idx // self.ipc
