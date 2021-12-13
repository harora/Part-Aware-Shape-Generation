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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--views', type=int, default=4)
parser.add_argument('--angles', type=int, default=3)
parser.add_argument('--batchsize', type=int, default=12)
parser.add_argument('--imgdir', type=str, default="single_pcd")
parser.add_argument('--checkpointdir', type=str, default="scorer/checkpoint")


def runPredictions(views, inputdir, imgdir, batch_size):
    classes = ['negative', 'positive']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    modelpath = os.path.join(inputdir, "model.pth")
    # dataset = ChairDataset()
    dataset = MultiviewImgDataset(imgdir, imgset="test", num_views=views, shuffle=False)
    noOfchairs = len(dataset)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, noOfchairs), num_workers=0, drop_last=True, shuffle=True)
    model = MVCNN(num_views=views).to(device)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    predictions = np.zeros((0, 1))
    scores = np.zeros((0, 2))
    filenames = []
    for images, labels, fnames in tqdm(dataloader):

        images = images.float().to(device)
        batch_size, num_views, c, w, h = images.shape

        pred_scores = model(images.reshape(batch_size*num_views, c, w, h))
        pred = torch.argmax(pred_scores, axis=1)
        pred = np.expand_dims(pred.detach().cpu().numpy(),axis=1)
        predictions = np.concatenate([predictions,pred], axis=0)
        pred_scores = pred_scores.detach().cpu().numpy()
        scores = np.concatenate([scores,pred_scores], axis=0)
        filenames.extend(fnames[0])
    predictions = predictions.astype(int)
    predictions = predictions.squeeze(axis=1)
    results = []
    for i in range(0, predictions.shape[0]):
        results.append((filenames[i], classes[predictions[i]], scores[i]))

    results = pd.DataFrame(results, columns=["Image name", "Predicted Class", "Scores"])
    results.to_csv(os.path.join(imgdir, "results.csv"))
    print("Done")

if __name__ == "__main__":
    args = parser.parse_args()
    views=args.views
    angles=args.angles
    checkpointdir=args.checkpointdir
    imgdir=args.imgdir
    batch_size = args.batchsize
    runPredictions(views*angles,checkpointdir, imgdir, batch_size)