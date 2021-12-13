import os
import mcubes
import numpy as np
import open3d as o3d
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance
from torch.utils.tensorboard import SummaryWriter

from model import Assembler
from data import ChairDataset, collate_fn
from pointnet2_cls_ssg import get_model_encoder
from utils import farthest_point_sample, compute_sampling_metrics, pc_normalize_sphere, assemble_from_multiple_batch, assemble_chairs

def voxel_grid_to_sparse(voxel_grid):
    voxels = voxel_grid.get_voxels()
    coords = []
    for voxel in voxels:
        coord = np.array(voxel.grid_index).astype(np.int32)
        coords.append(coord)
    coords = np.array(coords)
    return coords

def sparse_to_volume(coords):
    model = np.zeros(list(coords.max(0)+1))
    for i in range(coords.shape[0]):
        coord = coords[i]
        x, y, z = coord[0], coord[1], coord[2]
        model[x, y, z] = 1
    return model

def show_chairs(chairs):
    for i in range(chairs.shape[0]):
        pc = chairs[i]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([pcd])

hidden_dim = 256

pointnet2 = get_model_encoder(hidden_dim).cuda()
assembler = Assembler(hidden_dim).cuda()

pointnet2.load_state_dict(torch.load('./checkpoints_gan/pointnet2_13.pth'))
assembler.load_state_dict(torch.load('./checkpoints_gan/assembler_13.pth'))

############################################################################################################################################################################
"""
Set A
"""
bs = 5
num_workers = 8
val_dir = '../dataset/SetC'
val_dataset = ChairDataset(val_dir)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, drop_last=False, shuffle=True)

count = 0
final_eval = {}
pointnet2.eval()
assembler.eval()
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        batch_size = batch['mask'].shape[0]
        print(batch_size)

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

        fake_pc = assemble_chairs(parts, mask, pred).detach().cpu().numpy()
        show_chairs(fake_pc)
        print(fake_pc.shape)
        exit()

############################################################################################################################################################################
