import numpy as np

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
