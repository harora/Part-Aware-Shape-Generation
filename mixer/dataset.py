from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import json
import random
import trimesh



class PartNetDataset(Dataset):
    def __init__(self, phase, data_root, category, n_pts):
        super(PartNetDataset, self).__init__()
        if phase == "validation":
            phase = "val"

        self.phase = phase
        self.aug = phase == "train"

        self.data_root = data_root

        shape_names = collect_data_id(SPLIT_DIR, category, phase)
        self.shape_names = []
        for name in shape_names:
            path = os.path.join(PC_MERGED_LABEL_DIR, name)
            if os.path.exists(path):
                self.shape_names.append(name)

        self.n_pts = n_pts
        self.raw_n_pts = self.n_pts // 2

        self.rng = random.Random(1234)

    @staticmethod
    def load_point_cloud(path):
        pc = trimesh.load(path)
        pc = pc.vertices / 2.0 # scale to unit sphere
        return pc

    @staticmethod
    def read_point_cloud_part_label(path):
        with open(path, 'r') as fp:
            labels = fp.readlines()
        labels = np.array([int(x) for x in labels])
        return labels

    def random_rm_parts(self, raw_pc, part_labels):
        part_ids = sorted(np.unique(part_labels).tolist())
        if self.phase == "train":
            random.shuffle(part_ids)
            n_part_keep = random.randint(1, max(1, len(part_ids) - 1))
        else:
            self.rng.shuffle(part_ids)
            n_part_keep = self.rng.randint(1, max(1, len(part_ids) - 1))
        part_ids_keep = part_ids[:n_part_keep]
        point_idx = []
        for i in part_ids_keep:
            point_idx.extend(np.where(part_labels == i)[0].tolist())
        raw_pc = raw_pc[point_idx]
        return raw_pc, n_part_keep