from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
from skimage import io
import torchvision.transforms as T

# Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class ImgTextDataset(Dataset):
    imgs_frame: pd.DataFrame
    imgs_dir: str

    def __init__(self, csv_file: str = "labels.csv", imgs_dir: str = "./imgs", transform=None) -> None:
        super().__init__()

        self.imgs_frame = pd.read_csv(csv_file)
        self.imgs_dir = imgs_dir
        self.transform = transform

        print(len(self.imgs_frame))

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
    

def custom_collate_fn(batch):
    # 'batch' is a list of tuples (image, caption)
    print("got here")
    captions, images = zip(*batch)
    captions_list = list(captions)
    return captions_list, images

if __name__ == '__main__':
    dataset = ImgTextDataset(csv_file='./labels.csv', imgs_dir='./imgs', transform=T.ToTensor())

    dataloader = DataLoader(dataset=dataset, batch_size=5, collate_fn=custom_collate_fn)

    for i, sample in enumerate(dataset):
        print(sample[1].shape)
        pass