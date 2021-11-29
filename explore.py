import os
import json
import random
import numpy as np
import open3d as o3d
from glob import glob

semantic_mapping = {'chair_back': 0, 'chair_seat': 1, 'chair_base': 2, 'chair_arm': 3}
color_map = {0: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)), 1: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
             2: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)), 3: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))}

dataset_dir = './dataset/Chair_parts/*'
chair_dirs = glob(dataset_dir)

def dfs(node):
    if 'children' in node:
        leaf = []
        length = len(node['children'])
        for i in range(length):
            leaf = leaf + dfs(node['children'][i])
    else:
        # print(node)
        leaf = [node['id']]
    return leaf

for chair_dir in chair_dirs:
    local_map = {}
    ply_file = chair_dir + '/point_sample/ply-10000.ply'
    text_file = chair_dir + '/point_sample/label-10000.txt'
    # ply_file = chair_dir + '/point_sample/sample-points-all-pts-label-10000.ply'

    with open(text_file) as f:
        labels = f.readlines()
        labels = [int(line.rstrip()) for line in labels]
    labels = np.array(labels)
    print(labels.shape)
    print(np.unique(labels))

    with open(chair_dir + '/result.json') as f:
        hierarchy = json.load(f)
    hierarchy = hierarchy[0]
    semantic_parts = hierarchy['children']
    for semantic_part in semantic_parts:
        child_idx = dfs(semantic_part)
        # print(semantic_part['name'])
        # print(semantic_mapping[semantic_part['name']])
        # print(child_idx)
        # print()
        for idx in child_idx:
            local_map[idx] = semantic_mapping[semantic_part['name']]

    label_colors = []
    for i in range(labels.shape[0]):
        # label_colors.append(local_map[labels[i]])
        label_colors.append(color_map[local_map[labels[i]]])
    label_colors = np.array(label_colors)

    pcd = o3d.io.read_point_cloud(ply_file)
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    o3d.visualization.draw_geometries([pcd])

    # http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
    # pcd.estimate_normals()
    # radii = [0.025, 0.05, 0.1, 0.2]
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    # pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([pcd, rec_mesh])

    # print(ply_file)

    # print(np.asarray(pcd.points))
    # print(np.asarray(pcd.colors))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.035)
    o3d.visualization.draw_geometries([voxel_grid])
    # exit()
