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

def assemble(parts, mask, transformations):
    pc = []
    count = 0
    for i in range(5):
        if mask[i] == 1:
            part = parts[count]
            count += 1

            scale = transformations[i][:3].reshape(1, -1)
            trans = transformations[i][3:].reshape(1, -1)

            part = part * scale
            part = part + trans

            pc.append(part)
    pc = np.concatenate(pc, 0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

bs = 1
num_workers = 8
hidden_dim = 256

pointnet2 = get_model_encoder(hidden_dim).cuda()
assembler = Assembler(hidden_dim).cuda()

pointnet2.load_state_dict(torch.load('./checkpoints/pointnet2_19.pth'))
assembler.load_state_dict(torch.load('./checkpoints/assembler_19.pth'))

val_dir = '../dataset/processed_pcd/val'
val_dataset = ChairDataset(val_dir)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, drop_last=False, shuffle=True)

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

        assemble(batch['parts'].numpy(), batch['mask'].squeeze(0).numpy(), pred.squeeze(0).detach().cpu().numpy())
