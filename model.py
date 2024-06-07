import os
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import pandas as pd


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        fc_inputs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.resnet(x)

class MyDataset(Dataset):
    def __init__(self, width, height):
        super(MyDataset, self).__init__()
        self.width = width
        self.height = height
        self.labelPath = "/Users/suki/Downloads/label.txt"
        self.imgRoot = "/Users/suki/Downloads/20240531"
        self.transforms = transforms.Compose([
            transforms.Resize((self.width, self.height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        fileList = os.listdir(self.imgRoot)
        self.imgList = [imgName for imgName in fileList if imgName.endswith('jpg')]
        fileLabel = pd.read_csv("/Users/suki/Downloads/label.csv", sep=" ", header=None)
        self.label = {}
        for i in range(len(fileLabel[0])):
            self.label[fileLabel[0][i]] = fileLabel[1][i]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imgRoot, self.imgList[idx])).convert("RGB")
        img = self.transforms(img)
        label = self.label[self.imgList[idx]]
        return img, label

    def __len__(self):
        return len(self.imgList)


if __name__ == '__main__':
    dataset = MyDataset(512, 512)
    print(dataset[0])