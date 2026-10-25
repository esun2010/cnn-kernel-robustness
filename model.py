import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import medmnist
from medmnist import INFO

import tqdm
import math

from config import BATCH_SIZE, DISPLAY_INFO, IMAGE_RESOLUTION

#miscellaneous information about the dataset
def misc_info(display = DISPLAY_INFO):
    info = INFO["pathmnist"]
    if display:
        print("Version:" + torch.__version__)
        print("CUDA Available: " + str(torch.cuda.is_available()))

        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        print("Task: " + task)
        print("Number of Channels: " + str(n_channels))
        print("Number of Classes: " + str(n_classes))
    return info

#CNN model definition
class CNN(nn.Module):
    def __init__(self, kernel_sizes=[3, 3, 3]):
        super(CNN, self).__init__()
        info = misc_info(display = False)
        in_channels = info['n_channels']
        out_channels = len(info['label'])
        padding = [0,0,0]
        for i in range(len(kernel_sizes)):
            padding[i] = (kernel_sizes[i]-1)//2

        self.Layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=kernel_sizes[0], stride=1, padding=padding[0]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels, 32, kernel_size=kernel_sizes[0], stride=1, padding=padding[0]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Layer2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=kernel_sizes[1], stride=1, padding=padding[1]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels, 32, kernel_size=kernel_sizes[1], stride=1, padding=padding[1]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Layer3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=kernel_sizes[2], stride=1, padding=padding[2]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels, 32, kernel_size=kernel_sizes[2], stride=1, padding=padding[2]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(32*1*1,16),
            nn.ReLU(),
            nn.Linear(16, out_channels)
        )    

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_data_loader(split, size=IMAGE_RESOLUTION, blur = False, sigma=float(0)):
    info = misc_info(display = False)
    DataClass = getattr(medmnist, info['python_class'])
    if split == 0:
        split = "train"
    elif split == 1:
        split = "val"
    elif split == 2:
        split = "test"
    if blur:
        transform_cmds = transforms.Compose([
            transforms.ToTensor(),
            transforms.GaussianBlur(2*math.ceil(3*sigma)+1, sigma),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        transform = transform_cmds
    else:
        transform_cmds = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        transform = transform_cmds
    data = DataClass(split=split, transform=transform, download=True, size=size, mmap_mode="r")
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

