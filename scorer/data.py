import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ChairDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.positive_dir = '../dataset/LeChairs/chairs-data/positive'
        self.negative_dir = '../dataset/LeChairs/chairs-data/negative'
        self.positive_items = glob(self.positive_dir+'/*')
        self.negative_items = glob(self.negative_dir+'/*')
        self.offset = len(self.positive_items) // 3

    def __len__(self):
        return len(self.positive_items) // 3 + len(self.negative_items) // 3

    def __getitem__(self, idx):
        if idx >= self.offset:
            views = []
            for i in range(3):
                view_dir = self.negative_dir + '/' + '{:06d}.bmp'.format(3*(idx-self.offset)+i+1)
                img = cv2.imread(view_dir)
                views.append(img)

            label = 0
            views = np.array(views)
        else:
            views = []
            for i in range(3):
                view_dir = self.positive_dir + '/' + '{:06d}.bmp'.format(3*(idx)+i+1)
                img = cv2.imread(view_dir)
                views.append(img)

            label = 1
            views = np.array(views)

        return views.transpose(0, 3, 1, 2)/255, label

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = ChairDataset()
    dataloader = train_loader = DataLoader(dataset, batch_size=32, num_workers=1, drop_last=True, shuffle=True)

    for img, label in tqdm(dataloader):
        print(img.shape)
        print(label)

        temp = img[0, 0].permute(1,2,0).cpu().numpy()
        plt.imshow(temp)
        plt.show()

        temp = img[0, 1].permute(1,2,0).cpu().numpy()
        plt.imshow(temp)
        plt.show()

        temp = img[0, 2].permute(1,2,0).cpu().numpy()
        plt.imshow(temp)
        plt.show()
        exit()
