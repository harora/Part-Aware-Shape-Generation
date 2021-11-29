from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MVCNN
from data import ChairDataset

num_epochs = 20
batch_size = 32

dataset = ChairDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=True)

model = MVCNN(num_views=3).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epochs):
    counter = 0
    total_loss = 0
    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()

        images = images.float().cuda()
        labels = labels.cuda()
        batch_size, num_views, c, w, h = images.shape

        pred = model(images.reshape(batch_size*num_views, c, w, h))
        loss = F.nll_loss(pred, labels)
        counter += 1
        total_loss += loss.item()
        optimizer.step()
    total_loss /= counter
    print(total_loss)
