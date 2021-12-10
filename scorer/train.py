from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import argparse
from model import MVCNN
from data import ChairDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# parser = argparse.ArgumentParser()
# parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="mvcnn")

outputdir="checkpoint"
os.makedirs(outputdir, exist_ok=True)

num_epochs = 20
batch_size = 32

dataset = ChairDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)

model = MVCNN(num_views=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epochs):
    counter = 0
    total_loss = 0
    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()

        images = images.float().to(device)
        labels = labels.to(device)
        batch_size, num_views, c, w, h = images.shape

        pred = model(images.reshape(batch_size*num_views, c, w, h))
        loss = F.nll_loss(pred, labels)
        counter += 1
        total_loss += loss.item()
        optimizer.step()
    total_loss /= counter
    print(total_loss)

model.to("cpu")
torch.save(model.state_dict(), os.path.join(outputdir, "model.pth"))
print("done")
