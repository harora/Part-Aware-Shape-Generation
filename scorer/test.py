import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ImgDataset import *
from model import MVCNN
from data import ChairDataset
device = "cuda" if torch.cuda.is_available() else "cpu"
views=4
batch_size = 12
inputdir="scorer/checkpoint"
imgdir="scorer/chair"

# dataset = ChairDataset()
dataset = MultiviewImgDataset(imgdir, imgset="test", num_views=views, shuffle=False)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)

model = MVCNN(num_views=views).to(device)
model.train()
predictions = []
for images, labels, fnames in tqdm(dataloader):

    images = images.float().to(device)
    labels = labels.to(device)
    batch_size, num_views, c, w, h = images.shape

    pred = model(images.reshape(batch_size*num_views, c, w, h))
    predictions.append(pred)


print("Done")