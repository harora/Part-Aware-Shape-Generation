from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import argparse
import numpy as np
from model import MVCNN
from data import ChairDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# parser = argparse.ArgumentParser()
# parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="mvcnn")

outputdir="checkpoint"
os.makedirs(outputdir, exist_ok=True)

num_epochs = 10
batch_size = 18

dataset = ChairDataset()
print(f"Dataset len is {len(dataset)}")
train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size])

print(f"Train dataset len is {len(train_set)}, val set len is {len(val_set)}")


train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=False)

model = MVCNN(num_views=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
training_acc = []
val_acc = []
for epoch in range(num_epochs):
    counter = 0
    total_loss = 0

    for images, labels in tqdm(train_dataloader):
        optimizer.zero_grad()

        images = images.float().to(device)
        labels = labels.to(device)
        batch_size, num_views, c, w, h = images.shape

        pred = model(images.reshape(batch_size*num_views, c, w, h))
        loss = F.nll_loss(pred, labels)
        loss.backward()
        counter += 1
        total_loss += loss.item()
        optimizer.step()
    right_class = 0
    wrong_class = 0
    for images, labels in tqdm(val_dataloader):
        images = images.float().to(device)
        labels = labels.to(device)
        batch_size, num_views, c, w, h = images.shape
        pred = model(images.reshape(batch_size*num_views, c, w, h))
        pred = torch.argmax(pred, axis=1)
        rights = len(torch.where(pred == labels)[0])
        wrongs = labels.shape[0] - rights
        right_class += rights
        wrong_class = wrongs
    
        
    total_loss /= counter
    val = rights / (rights+wrongs)
    validation_acc = round((val)*100, 3)
    print(f"Epoch loss is {total_loss}, Validation acc is {validation_acc}%" )
    training_acc.append(total_loss)
    val_acc.append(val)

training_acc = np.array(training_acc)
val_acc = round(np.array(val_acc).mean()*100, 3)
model.to("cpu")
torch.save(model.state_dict(), os.path.join(outputdir, "model.pth"))
print(f"Done. Mean training acc is {training_acc.mean()}, mean validation acc {val_acc}")
