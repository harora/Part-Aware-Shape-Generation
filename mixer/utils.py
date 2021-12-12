import numpy as np
import open3d as o3d
from multiprocessing import Pool

import torch
from torch.utils.data._utils.collate import default_collate

def assemble_chairs(all_parts, all_mask, all_transformations):
    batch_size = all_mask.shape[0]
    offsets = all_mask.reshape(batch_size, -1).sum(-1).long()
    
    offset = 0

    fake_pc = []
    for bs in range(batch_size):
        parts = all_parts[offset:offsets[:bs+1].sum()].permute(0, 2, 1)
        offset = offsets[:bs+1].sum()

        mask = all_mask[bs]
        transformations = all_transformations[bs]

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
        pc = torch.cat(pc)
        sample_idx = farthest_point_sample_idx(pc.detach().cpu().numpy(), 2048)
        pc = pc[sample_idx]
        fake_pc.append(pc.unsqueeze(0))
    fake_pc = torch.cat(fake_pc, 0)
    return fake_pc

def assemble_from_multiple_batch(chairs):
    batch_size = chairs['mask'].shape[0]
    copy_input = [chairs for i in range(batch_size)]
    with Pool(8) as p:
        r = list(p.imap(assemble_from_multiple, copy_input))

    def collate_fn(batch):
        default_collate_items = ['mask']

        data = []
        all_parts = []
        for item in batch:
            all_parts.append(item['parts'])
            data.append({k:item[k] for k in default_collate_items})
        data = default_collate(data)
        data['parts'] = torch.cat(all_parts, 0)
        return data

    fake_data = collate_fn(r)

    return fake_data
    
def assemble_from_multiple(chairs):
    np.random.seed()
    batch_size = chairs['mask'].shape[0]
    select_idx = (chairs['mask'] == 1).nonzero(as_tuple=False)
    select_idx = select_idx[:, 0] * 5 + select_idx[:, 1]
    parts_padded = torch.zeros((batch_size, 5, chairs['parts'].shape[1], chairs['parts'].shape[2]))
    parts_padded = parts_padded.reshape(-1, chairs['parts'].shape[1], chairs['parts'].shape[2])
    parts_padded[select_idx] = chairs['parts'].float()
    parts_padded = parts_padded.reshape(batch_size, 5, chairs['parts'].shape[1], chairs['parts'].shape[2])

    new_parts = []
    new_part_mask = np.zeros(5)
    for i in range(3):
        distribution = chairs['mask'][:, i]
        distribution = distribution / distribution.sum()

        chosen_idx = np.random.choice(np.arange(batch_size), p=distribution)
        new_parts.append(parts_padded[chosen_idx, i].unsqueeze(0))
        new_part_mask[i] = 1

    arm_rest_distribution = chairs['mask'][:, 4]
    null_prob = torch.zeros(1)
    null_prob[0] = 2
    arm_rest_distribution = torch.cat((arm_rest_distribution, null_prob))
    arm_rest_distribution = arm_rest_distribution / arm_rest_distribution.sum()
    chosen_idx = np.random.choice(np.arange(batch_size+1), p=arm_rest_distribution)

    if chosen_idx != batch_size:
        new_parts.append(parts_padded[chosen_idx, 3].unsqueeze(0))
        new_part_mask[3] = 1

        new_parts.append(parts_padded[chosen_idx, 4].unsqueeze(0))
        new_part_mask[4] = 1
    new_parts = torch.cat(new_parts, 0)

    return {'parts': new_parts, 'mask': new_part_mask}

def pc_normalize_sphere(pc):
    """
    From https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ModelNetDataLoader.py

    Translates pc to have 0 mean and scales to within unit sphere
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    m = np.array(m).reshape(1)
    return pc, np.concatenate((m, centroid), axis=0)

def pc_normalize_cube(pc):
    """
    From https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ModelNetDataLoader.py

    Translates pc to have 0 mean and scales to within unit cube
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(pc, axis=0)
    pc = pc / m
    return pc, np.concatenate((m, centroid), axis=0)

def farthest_point_sample(point, npoint):
    """
    From https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ModelNetDataLoader.py

    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def farthest_point_sample_idx(point, npoint):
    """
    From https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ModelNetDataLoader.py

    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype(np.int32)

