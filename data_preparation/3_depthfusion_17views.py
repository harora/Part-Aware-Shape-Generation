import numpy as np
import cv2
import os
import binvox_rw_customized
#import mcubes
import cutils
import argparse
from sort_nicely import *


parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", type=str, help="path to partnet dataset")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_args()

dataset_path = FLAGS.dataset_path
target_root = dataset_path + "/"
if not os.path.exists(target_root):
    print("ERROR: this dir does not exist: " + target_root)
    exit()

dataset_path = FLAGS.dataset_path
target_root = dataset_path + "/"
if not os.path.exists(target_root):
    print("ERROR: this dir does not exist: " + target_root)
    exit()


share_id = FLAGS.share_id
share_total = FLAGS.share_total

shape_names = sort_nicely(os.listdir(target_root))

start = int(share_id*len(shape_names)/share_total)
end = int((share_id+1)*len(shape_names)/share_total)
shape_names = shape_names[start:end]


def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
    fout.close()


rendering = np.zeros([320,320,17], np.int32)
state_ctr = np.zeros([64*64*64,2], np.int32)


for shape_name in shape_names:   
    shape_root = target_root + shape_name
    if not os.path.isdir(shape_root):
        continue
    voxels_root = shape_root + "/voxels/"

    filled_voxels_root = shape_root + "/filled_voxels/"
    if not os.path.exists(filled_voxels_root):
	    os.makedirs(filled_voxels_root)

    for binvox_name in sort_nicely(os.listdir(voxels_root)):
        if not binvox_name.endswith(".binvox"):
            continue
        voxel_name = voxels_root + binvox_name
        out_name = filled_voxels_root + binvox_name.split('.')[0] + "_filled.binvox"
        print(voxel_name)

        voxel_model_file = open(voxel_name, 'rb')
        vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file,fix_coords=False)

        batch_voxels = vox_model.data.astype(np.uint8)
        rendering[:] = 2**16
        cutils.depth_fusion_XZY(batch_voxels,rendering,state_ctr)


        with open(out_name, 'wb') as fout:
            binvox_rw_customized.write(vox_model, fout, state_ctr)

        '''
        voxel_model_file = open(out_name, 'rb')
        vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file)
        batch_voxels = vox_model.data.astype(np.uint8)
        vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
        write_ply_triangle("vox.ply", vertices, triangles)

        for j in range(17):
            img = (rendering[:,:,j]<2**16).astype(np.uint8)*255
            cv2.imwrite("vox_"+str(j)+".png", img)

        exit(0)
        '''
    

