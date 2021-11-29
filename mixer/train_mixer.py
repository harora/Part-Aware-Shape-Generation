import os
import glob
import numpy as np

from torch.utils.data import Dataset, DataLoader

from dataset import PartNetDataset



# Setup Data

dataset = PartNetDataset(phase='train',data_root='/scratch/data_v0/')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4,
                            worker_init_fn=np.random.seed())
# Setup Network

#encoder (MLP), Generator(transformations), Discriminator



# Train


for b, data in enumerate(dataloader):
    print(data)

    break