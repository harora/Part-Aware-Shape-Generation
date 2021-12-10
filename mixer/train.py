import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Assembler
from data import ChairDataset, collate_fn
from pointnet2_cls_ssg import get_model_encoder

checkpoints_directory = './checkpoints'
if not os.path.exists(checkpoints_directory):
    os.makedirs(checkpoints_directory)

# Settings
epochs = 20
bs = 8
num_workers = 16
hidden_dim = 256

# Get train dataloader
train_dir = '../dataset/processed_pcd/train'
train_dataset = ChairDataset(train_dir)
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, drop_last=True, shuffle=True)

# Get val dataloader
val_dir = '../dataset/processed_pcd/val'
val_dataset = ChairDataset(val_dir)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, drop_last=False, shuffle=True)

# Get models and optimizer
pointnet2 = get_model_encoder(hidden_dim).cuda()
assembler = Assembler(hidden_dim).cuda()
optimizer = optim.Adam(list(pointnet2.parameters()) + list(assembler.parameters()))

writer = SummaryWriter()

l1 = nn.L1Loss()
for epoch in range(epochs):
    count = 0
    total_train_loss = 0
    pointnet2.train()
    assembler.train()
    for batch in tqdm(train_dataloader):
        mask = batch['mask'].float().cuda().unsqueeze(-1)
        parts = batch['parts'].permute(0, 2, 1).float().cuda()
        transformations = batch['transformations'].float().cuda()
        batch_size = transformations.shape[0]

        optimizer.zero_grad()
        part_f, _ = pointnet2(parts)
        obj_f = torch.zeros((batch_size, 5, hidden_dim)).cuda()

        select_idx = (batch['mask'] == 1).nonzero(as_tuple=False)
        select_idx = select_idx[:, 0] * 5 + select_idx[:, 1]
        obj_f = obj_f.reshape(-1, hidden_dim)
        obj_f[select_idx] = part_f
        obj_f = obj_f.reshape(batch_size, 5, hidden_dim)

        obj_f = obj_f.reshape(batch_size, -1)
        pred = assembler(obj_f)

        recon_loss = l1(pred * mask, transformations * mask)
        loss = recon_loss
        loss.backward()
        optimizer.step()

        count += 1
        total_train_loss += loss.item()
    total_train_loss /= count
    writer.add_scalar('Loss/train', total_train_loss, epoch)

    count = 0
    total_val_loss = 0
    pointnet2.eval()
    assembler.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            mask = batch['mask'].float().cuda().unsqueeze(-1)
            parts = batch['parts'].permute(0, 2, 1).float().cuda()
            transformations = batch['transformations'].float().cuda()
            batch_size = transformations.shape[0]

            part_f, _ = pointnet2(parts)
            obj_f = torch.zeros((batch_size, 5, hidden_dim)).cuda()

            select_idx = (batch['mask'] == 1).nonzero(as_tuple=False)
            select_idx = select_idx[:, 0] * 5 + select_idx[:, 1]
            obj_f = obj_f.reshape(-1, hidden_dim)
            obj_f[select_idx] = part_f
            obj_f = obj_f.reshape(batch_size, 5, hidden_dim)

            obj_f = obj_f.reshape(batch_size, -1)
            pred = assembler(obj_f)

            recon_loss = l1(pred * mask, transformations * mask)
            loss = recon_loss

            count += 1
            total_val_loss += loss.item()
    total_val_loss /= count
    writer.add_scalar('Loss/val', total_val_loss, epoch)

    torch.save(pointnet2.state_dict(), checkpoints_directory+'/pointnet2_{}.pth'.format(epoch))
    torch.save(assembler.state_dict(), checkpoints_directory+'/assembler_{}.pth'.format(epoch))
    print('Epoch: {}, Train Loss: {}, Val Loss: {}'.format(epoch, total_train_loss, total_val_loss))
