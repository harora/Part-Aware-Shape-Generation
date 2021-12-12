import os
import json
import shutil
import random
import numpy as np
import open3d as o3d
from glob import glob
from tqdm import tqdm

semantic_mapping = {'chair_back': 0, 'chair_seat': 1, 'chair_base': 2, 'chair_arm_1': 3, 'chair_arm_2': 4}
color_map = {0: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)), 1: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
             2: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)), 3: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
             4: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))}

dataset_dir = '../dataset/Chair_parts/*'
chair_dirs = glob(dataset_dir)

val_percentage = 0.05

save_directory = '../dataset/processed_pcd'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
if not os.path.exists('../dataset/processed_pcd/train'):
    os.makedirs('../dataset/processed_pcd/train')
if not os.path.exists('../dataset/processed_pcd/val'):
    os.makedirs('../dataset/processed_pcd/val')

# DFS to find what semantic node each leaf node belongs to
def dfs(node):
    if 'children' in node:
        leaf = []
        length = len(node['children'])
        for i in range(length):
            leaf = leaf + dfs(node['children'][i])
    else:
        leaf = [node['id']]
    return leaf

ignore_count = 0
processed_chairs = []
for chair_dir in tqdm(chair_dirs):
    local_map = {}
    chair_idx = chair_dir.split('/')[-1]
    ply_file = chair_dir + '/point_sample/ply-10000.ply'
    text_file = chair_dir + '/point_sample/label-10000.txt'

    # Read point clouds from the file, obtain the xyz
    pcd = o3d.io.read_point_cloud(ply_file)
    xyz = np.asarray(pcd.points)

    # These are leaf node ids
    with open(text_file) as f:
        labels = f.readlines()
        labels = [int(line.rstrip()) for line in labels]
    labels = np.array(labels)

    # The semantic parts are all from the second level of the tree
    with open(chair_dir + '/result.json') as f:
        hierarchy = json.load(f)
    hierarchy = hierarchy[0]
    semantic_parts = hierarchy['children']

    # Ignore shapes that have more than 2 chair arms or have "other" semantic parts
    arm_count = 0
    other_count = 0
    for semantic_part in semantic_parts:
        if semantic_part['name'] == 'chair_arm':
            arm_count += 1
        elif semantic_part['name'] == 'other':
            other_count += 1
    if arm_count > 2 or other_count > 0:
        ignore_count += 1
        continue

    # Map all leaf node ids to their respective second level semantic part nodes
    arm_count = 1
    for semantic_part in semantic_parts:
        child_idx = dfs(semantic_part)
        if semantic_part['name'] == 'chair_arm':
            for idx in child_idx:
                local_map[idx] = semantic_mapping[semantic_part['name'] + '_{}'.format(arm_count)]
            arm_count += 1
        else:
            for idx in child_idx:
                local_map[idx] = semantic_mapping[semantic_part['name']]

    # Get the remapped labels for each leaf node
    for i in range(labels.shape[0]):
        labels[i] = local_map[labels[i]]
    labels = labels.reshape(-1, 1)

    # Save the files as an compressed npz
    np.savez(save_directory+'/{}.npz'.format(chair_idx), xyz=xyz, labels=labels)
    processed_chairs.append(chair_idx)

random.shuffle(processed_chairs)
total_count = len(processed_chairs)
train_count = int(len(processed_chairs) * (1 - val_percentage))
val_count = len(processed_chairs) - train_count

dataset_json = {'train': [], 'val': []}
for i in range(train_count):
    shutil.move(save_directory+'/{}.npz'.format(processed_chairs[i]), save_directory+'/train/{}.npz'.format(processed_chairs[i]))
    dataset_json['train'].append(processed_chairs[i])

for i in range(val_count):
    shutil.move(save_directory+'/{}.npz'.format(processed_chairs[i+train_count]), save_directory+'/val/{}.npz'.format(processed_chairs[i+train_count]))
    dataset_json['val'].append(processed_chairs[i+train_count])

with open(save_directory+'/split.json', 'w') as fp:
    json.dump(dataset_json, fp)

print('Skipped {} shapes'.format(ignore_count))
