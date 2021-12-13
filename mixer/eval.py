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
from utils import farthest_point_sample, compute_sampling_metrics, pc_normalize_sphere

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
            part = part * 5

            pc.append(part)
    pc = np.concatenate(pc, 0)
    pc = farthest_point_sample(pc, 2048)
    pc, _ = pc_normalize_sphere(pc)
    pc = torch.from_numpy(pc).unsqueeze(0).float()
    return pc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.25)
    o3d.visualization.draw_geometries([voxel_grid])

    coords = voxel_grid_to_sparse(voxel_grid)
    volume = sparse_to_volume(coords)
    vertices, triangles = mcubes.marching_cubes(volume, 0)
    mcubes.export_obj(vertices, triangles, 'example.obj')
    # exit()

    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
    # o3d.visualization.draw_geometries([voxel_grid])

    # pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    # pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(15)
    
    # radii = [0.05, 0.1, 0.2, 0.4]
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([pcd, rec_mesh])

bs = 1
num_workers = 8
hidden_dim = 256

pointnet2 = get_model_encoder(hidden_dim).cuda()
assembler = Assembler(hidden_dim).cuda()

pointnet2.load_state_dict(torch.load('./checkpoints/pointnet2_13.pth'))
assembler.load_state_dict(torch.load('./checkpoints/assembler_13.pth'))

val_dir = '../dataset/processed_pcd/val'
val_dataset = ChairDataset(val_dir)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, drop_last=False, shuffle=True)

count = 0
final_eval = {}
pointnet2.eval()
assembler.eval()
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        count += 1
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

        fake_pc = assemble(batch['parts'].numpy(), batch['mask'].squeeze(0).numpy(), pred.squeeze(0).detach().cpu().numpy())
        real_pc, _ = pc_normalize_sphere(batch['real_pc'].squeeze(0).numpy())
        real_pc = torch.from_numpy(real_pc).unsqueeze(0).float()

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        item = compute_sampling_metrics(fake_pc * 5, None, real_pc * 5, None, thresholds, 1e-8)

        for k, v in item.items():
            if k in final_eval:
                final_eval[k] += v.item()
            else:
                final_eval[k] = v.item()

for k, v in final_eval.items():
    if k in ['Chamfer-L2', 'F1@0.100000', 'F1@0.200000', 'F1@0.300000', 'F1@0.400000', 'F1@0.500000']:
        print(k + ': {}'.format(v/count))
