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
import pandas as pd
classes = ['negative', 'positive']
device = "cuda" if torch.cuda.is_available() else "cpu"
views=6
batch_size = 12
inputdir="scorer/checkpoint"
imgdir="scorer/chair"
modelpath = os.path.join(inputdir, "model.pth")
# dataset = ChairDataset()
dataset = MultiviewImgDataset(imgdir, imgset="test", num_views=views, shuffle=False)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
noOfchairs = len(dataset)
model = MVCNN(num_views=views).to(device)
model.load_state_dict(torch.load(modelpath))
model.eval()
predictions = np.zeros((0, 1))
filenames = []
for images, labels, fnames in tqdm(dataloader):

    images = images.float().to(device)
    batch_size, num_views, c, w, h = images.shape

    pred = model(images.reshape(batch_size*num_views, c, w, h))
    pred = torch.argmax(pred, axis=1)
    pred = np.expand_dims(pred.detach().numpy(),axis=1)
    predictions = np.concatenate([predictions,pred], axis=0)
    filenames.append(fnames)
predictions = predictions.astype(int)
predictions = predictions.squeeze()
results = []
for i in range(0, predictions.shape[0]):
    results.append((filenames[i*views][0], classes[predictions[i]]))

results = pd.DataFrame([results], columns=["Image name", "Predicted Class"])
results.to_csv("scorer/results.csv")
print("Done")