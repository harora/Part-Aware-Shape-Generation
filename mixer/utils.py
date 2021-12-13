import time
import numpy as np
import open3d as o3d
from multiprocessing import Pool

import torch
from torch.utils.data._utils.collate import default_collate
from pytorch3d.ops import knn_gather, knn_points, sample_points_from_meshes

def compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, thresholds, eps):
    """
    Compute metrics that are based on sampling points and normals:
    - L2 Chamfer distance
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    - Normal consistency (if normals are provided)
    - Absolute normal consistency (if normals are provided)
    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - gt_normals: Tensor of shape (N, S, 3) giving normals of points sampled from
          the ground-truth verts, or None of such normals are not available
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation
    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
    """
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)
    if gt_normals is not None:
        pred_normals_near = knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]  # (N, S, 3)
    else:
        pred_normals_near = None

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    if pred_normals is not None:
        gt_normals_near = knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]  # (N, S, 3)
    else:
        gt_normals_near = None

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    metrics["Chamfer-L2"] = chamfer_l2

    # Compute normal consistency and absolute normal consistance only if
    # we actually got normals for both meshes
    if pred_normals is not None and gt_normals is not None:
        pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
        gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)

        pred_to_gt_cos_sim = pred_to_gt_cos.mean(dim=1)
        pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
        gt_to_pred_cos_sim = gt_to_pred_cos.mean(dim=1)
        gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
        normal_dist = 0.5 * (pred_to_gt_cos_sim + gt_to_pred_cos_sim)
        abs_normal_dist = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
        metrics["NormalConsistency"] = normal_dist
        metrics["AbsNormalConsistency"] = abs_normal_dist

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def assemble_chairs(all_parts, all_mask, all_transformations):
    batch_size = all_mask.shape[0]
    offsets = all_mask.reshape(batch_size, -1).sum(-1).long()
    
    offset = 0

    np_pc = []
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

        item = {}
        item['point'] = pc.detach().cpu().numpy()
        item['npoint'] = 2048
        np_pc.append(item)

        fake_pc.append(pc)

    with Pool(8) as p:
        sample_idx = list(p.imap(farthest_point_sample_idx, np_pc))
    fake_pc = [fake_pc[i][sample_idx[i]].unsqueeze(0) for i in range(len(sample_idx))]
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

def farthest_point_sample_idx(data):
    """
    From https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ModelNetDataLoader.py

    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    point = data['point']
    npoint = data['npoint']

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

