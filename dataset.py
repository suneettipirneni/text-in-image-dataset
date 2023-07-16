from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from skimage import io
from torchvision.transforms import ToTensor

# Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class ImgTextDataset(Dataset):
    imgs_frame: pd.DataFrame
    imgs_dir: str

    def __init__(self, csv_file: str = "labels.csv", imgs_dir: str = "./imgs", transform=None) -> None:
        super().__init__()

        self.imgs_frame = pd.read_csv(csv_file)
        self.imgs_dir = imgs_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgs_frame)
    
    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = os.path.join(self.imgs_dir, self.imgs_frame.iloc[index, 1])
        img = io.imread(img_path)

        caption = self.imgs_frame.iloc[index, 0]

        if self.transform:
          img = self.transform(img)

        return caption, img

if __name__ == '__main__':
    dataset = ImgTextDataset(csv_file='./labels.csv', imgs_dir='./imgs', transform=ToTensor())

    for i, sample in enumerate(dataset):
        print(sample)