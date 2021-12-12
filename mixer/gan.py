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
from pointnet2_cls_ssg import get_model, get_model_encoder
from utils import assemble_from_multiple_batch, assemble_chairs

checkpoints_directory = './checkpoints_gan'
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
optimizer_g = optim.Adam(list(pointnet2.parameters()) + list(assembler.parameters()))

pointnet2.load_state_dict(torch.load('./checkpoints/pointnet2_19.pth'))
assembler.load_state_dict(torch.load('./checkpoints/assembler_19.pth'))

discriminator = get_model(1, normal_channel=False).cuda()
optimizer_d = optim.Adam(discriminator.parameters())

writer = SummaryWriter()

l1 = nn.L1Loss()
for epoch in range(epochs):
    count = 0
    total_g_loss = 0
    total_d_loss = 0
    total_recon_loss = 0
    total_g_fake_loss = 0

    pointnet2.train()
    assembler.train()
    for batch in tqdm(train_dataloader):
        with torch.autograd.set_detect_anomaly(True):
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            #################################################################################################################
            """
            Calculate reconstruction loss
            """
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

            real_pc = assemble_chairs(parts, mask, transformations).permute(0, 2, 1)
            recon_loss = l1(pred * mask, transformations * mask) * 100
            #################################################################################################################

            #################################################################################################################
            """
            Calculate generator loss
            """
            fake_batch = assemble_from_multiple_batch(batch)
            mask = fake_batch['mask'].float().cuda().unsqueeze(-1)
            parts = fake_batch['parts'].permute(0, 2, 1).float().cuda()

            part_f, _ = pointnet2(parts)
            obj_f = torch.zeros((batch_size, 5, hidden_dim)).cuda()

            select_idx = (fake_batch['mask'] == 1).nonzero(as_tuple=False)
            select_idx = select_idx[:, 0] * 5 + select_idx[:, 1]
            obj_f = obj_f.reshape(-1, hidden_dim)
            obj_f[select_idx] = part_f
            obj_f = obj_f.reshape(batch_size, 5, hidden_dim)

            obj_f = obj_f.reshape(batch_size, -1)
            pred = assembler(obj_f)

            fake_pc = assemble_chairs(parts, mask, pred).permute(0, 2, 1)
            fake_logit, fake_feat = discriminator(fake_pc)

            g_loss = -fake_feat.mean() + recon_loss
            g_loss.backward()
            optimizer_g.step()
            #################################################################################################################

            #################################################################################################################
            """
            Calculate discriminator loss
            """
            fake_batch = assemble_from_multiple_batch(batch)
            mask = fake_batch['mask'].float().cuda().unsqueeze(-1)
            parts = fake_batch['parts'].permute(0, 2, 1).float().cuda()

            part_f, _ = pointnet2(parts)
            obj_f = torch.zeros((batch_size, 5, hidden_dim)).cuda()

            select_idx = (fake_batch['mask'] == 1).nonzero(as_tuple=False)
            select_idx = select_idx[:, 0] * 5 + select_idx[:, 1]
            obj_f = obj_f.reshape(-1, hidden_dim)
            obj_f[select_idx] = part_f
            obj_f = obj_f.reshape(batch_size, 5, hidden_dim)

            obj_f = obj_f.reshape(batch_size, -1)
            pred = assembler(obj_f)

            fake_pc = assemble_chairs(parts, mask, pred).permute(0, 2, 1)
            fake_logit, fake_feat = discriminator(fake_pc)
            real_logit, real_feat = discriminator(real_pc)

            d_loss = -real_feat.mean() + fake_feat.mean()
            d_loss.backward()
            optimizer_d.step()
            #################################################################################################################

            count += 1
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_recon_loss += recon_loss.item()
            total_g_fake_loss += -fake_feat.mean().item()
    total_g_loss /= count
    total_d_loss /= count
    total_recon_loss /= count
    total_g_fake_loss /= count
    writer.add_scalar('Loss/g_loss', total_g_loss, epoch)
    writer.add_scalar('Loss/d_loss', total_d_loss, epoch)
    writer.add_scalar('Loss/recon_loss', total_recon_loss, epoch)
    writer.add_scalar('Loss/g_fake_loss', total_g_fake_loss, epoch)

    count = 0
    total_g_loss = 0
    total_d_loss = 0
    total_recon_loss = 0
    total_g_fake_loss = 0

    pointnet2.eval()
    assembler.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            #################################################################################################################
            """
            Calculate reconstruction loss
            """
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

            real_pc = assemble_chairs(parts, mask, transformations).permute(0, 2, 1)
            recon_loss = l1(pred * mask, transformations * mask) * 100
            #################################################################################################################

            #################################################################################################################
            """
            Calculate generator loss
            """
            fake_batch = assemble_from_multiple_batch(batch)
            mask = fake_batch['mask'].float().cuda().unsqueeze(-1)
            parts = fake_batch['parts'].permute(0, 2, 1).float().cuda()

            part_f, _ = pointnet2(parts)
            obj_f = torch.zeros((batch_size, 5, hidden_dim)).cuda()

            select_idx = (fake_batch['mask'] == 1).nonzero(as_tuple=False)
            select_idx = select_idx[:, 0] * 5 + select_idx[:, 1]
            obj_f = obj_f.reshape(-1, hidden_dim)
            obj_f[select_idx] = part_f
            obj_f = obj_f.reshape(batch_size, 5, hidden_dim)

            obj_f = obj_f.reshape(batch_size, -1)
            pred = assembler(obj_f)

            fake_pc = assemble_chairs(parts, mask, pred).permute(0, 2, 1)
            fake_logit, fake_feat = discriminator(fake_pc)

            g_loss = -fake_feat.mean() + recon_loss
            #################################################################################################################

            #################################################################################################################
            """
            Calculate discriminator loss
            """
            real_logit, real_feat = discriminator(real_pc)

            d_loss = -real_feat.mean() + fake_feat.mean()
            #################################################################################################################

            count += 1
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_recon_loss += recon_loss.item()
            total_g_fake_loss += -fake_feat.mean().item()
    total_g_loss /= count
    total_d_loss /= count
    total_recon_loss /= count
    total_g_fake_loss /= count
    writer.add_scalar('Loss_val/g_loss', total_g_loss, epoch)
    writer.add_scalar('Loss_val/d_loss', total_d_loss, epoch)
    writer.add_scalar('Loss_val/recon_loss', total_recon_loss, epoch)
    writer.add_scalar('Loss_val/g_fake_loss', total_g_fake_loss, epoch)

    torch.save(pointnet2.state_dict(), checkpoints_directory+'/pointnet2_{}.pth'.format(epoch))
    torch.save(assembler.state_dict(), checkpoints_directory+'/assembler_{}.pth'.format(epoch))
    print('Epoch: {}, Train Loss: {}, Val Loss: {}'.format(epoch, total_train_loss, total_val_loss))
