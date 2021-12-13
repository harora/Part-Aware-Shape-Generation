import numpy as np
import open3d as o3d
from glob import glob

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from utils import pc_normalize_sphere, pc_normalize_cube, farthest_point_sample

def collate_fn(batch):
    default_collate_items = ['transformations', 'mask', 'real_pc']

    data = []
    all_parts = []
    for item in batch:
        all_parts.append(item['parts'])
        data.append({k:item[k] for k in default_collate_items})
    data = default_collate(data)
    data['parts'] = torch.from_numpy(np.concatenate(all_parts, axis=0))
    return data

class ChairDataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.chair_dirs = glob(dataset_dir+'/*.npz')
        self.part_name = {0: 'chair_back', 1: 'chair_seat', 2: 'chair_base', 3: 'chair_arm_1', 4: 'chair_arm_2'}

    def __len__(self):
        return len(self.chair_dirs)

    def __getitem__(self, idx):
        # Load xyz and part labels from dir
        data = np.load(self.chair_dirs[idx])
        xyz = data['xyz']
        labels = data['labels']

        # Subsample entire point cloud
        real_pc = farthest_point_sample(xyz, 2048)

        # Get the part labels the object has
        unique_labels = np.unique(labels)
        unique_labels = sorted(list(unique_labels))

        # Get each separate part and subsample
        parts = []
        part_mask = np.zeros(5)
        transformations = np.zeros((5, 6))
        for i in range(5):
            if i in unique_labels:
                # Get the part mask and extract
                mask = labels == i
                mask = list(mask.reshape(-1))
                part = xyz[mask]

                # Subsample to have 2048 point per part
                part = farthest_point_sample(part, 2048)
                # Normalize and get transformations
                part, transformation = pc_normalize_cube(part)

                # Store the part mask and other info
                part_mask[i] = 1
                parts.append(part)
                transformations[i] = transformation

        # Printing and visualization for debugging
        # print(xyz.shape)
        # print(labels.shape)
        # print(unique_labels)
        # print(part_mask)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # o3d.visualization.draw_geometries([pcd])
        # for idx, part in enumerate(parts):
        #     print(part_name[idx])
        #     print(part.shape)
        #     print(np.mean(part))
        #     print(np.max(part, axis=0))
        #     print()
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(part)
        #     o3d.visualization.draw_geometries([pcd])
            
        return {'parts': parts, 'transformations': transformations, 'mask': part_mask, 'unique_labels': unique_labels, 'real_pc': real_pc}

if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    batch_size = 8
    num_workers = 16
    train_dir = '../dataset/processed_pcd/train'
    train_dataset = ChairDataset(train_dir)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)

    max_val = 0
    max_val2 = 0
    for batch in tqdm(train_dataloader):
        if np.max(batch['transformations'][:, :, :3].numpy()) > max_val:
            max_val = np.max(batch['transformations'][:, :, :3].numpy())

        if np.max(batch['transformations'][:, :, 3:].numpy()) > max_val2:
            max_val2 = np.max(batch['transformations'][:, :, 3:].numpy())
    print(max_val)
    print(max_val2)
