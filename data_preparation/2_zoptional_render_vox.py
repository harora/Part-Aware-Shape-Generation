import numpy as np
import cv2
import os
import sys
import binvox_rw
from sort_nicely import *


target_root = sys.argv[1] + "/"
if not os.path.exists(target_root):
	print("ERROR: this dir does not exist: " + target_root)
	exit()

depths_root = "../depth_render/"
if not os.path.exists(depths_root):
	os.makedirs(depths_root)


for shape_name in sort_nicely(os.listdir(target_root)):   
    shape_root = target_root + shape_name
    if not os.path.isdir(shape_root):
        continue
    voxels_root = shape_root + "/voxels/"

    shape_depth_root = depths_root + "/" + shape_name + "/"
    if not os.path.exists(shape_depth_root):
	    os.makedirs(shape_depth_root)

    for binvox_name in sort_nicely(os.listdir(voxels_root)):
        if not binvox_name.endswith(".binvox"):
            continue
        voxel_name = voxels_root + binvox_name
        write_dir1 = shape_depth_root + binvox_name.split('.')[0] + ".png"
        print(voxel_name)

        voxel_model_file = open(voxel_name, 'rb')
        batch_voxels = binvox_rw.read_as_3d_array(voxel_model_file).data.astype(np.uint8)

        out = np.zeros([512*2,512*4], np.uint8)

        tmp = batch_voxels
        mask = np.amax(tmp, axis=0).astype(np.int32)
        depth = np.argmax(tmp,axis=0)
        depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
        depth = depth*mask
        out[512*0:512*1,512*0:512*1] = depth[::-1,:]

        mask = np.amax(tmp, axis=1).astype(np.int32)
        depth = np.argmax(tmp,axis=1)
        depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
        depth = depth*mask
        out[512*0:512*1,512*1:512*2] = depth

        mask = np.amax(tmp, axis=2).astype(np.int32)
        depth = np.argmax(tmp,axis=2)
        depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
        depth = depth*mask
        out[512*0:512*1,512*2:512*3] = np.transpose(depth)[::-1,::-1]

        tmp = batch_voxels[::-1,:,:]
        mask = np.amax(tmp, axis=0).astype(np.int32)
        depth = np.argmax(tmp,axis=0)
        depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
        depth = depth*mask
        out[512*1:512*2,512*0:512*1] = depth[::-1,::-1]
        redisual = np.clip(np.abs(mask[:,:] - mask[:,::-1])*256,0,255)
        out[512*0:512*1,512*3:512*4] = redisual[::-1,::-1]

        tmp = batch_voxels[:,::-1,:]
        mask = np.amax(tmp, axis=1).astype(np.int32)
        depth = np.argmax(tmp,axis=1)
        depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
        depth = depth*mask
        out[512*1:512*2,512*1:512*2] = depth[:,::-1]
        redisual = np.clip(np.abs(mask[:,:] - mask[:,::-1])*256,0,255)
        out[512*1:512*2,512*3:512*4] = redisual[:,::-1]

        tmp = batch_voxels[:,:,::-1]
        mask = np.amax(tmp, axis=2).astype(np.int32)
        depth = np.argmax(tmp,axis=2)
        depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
        depth = depth*mask
        out[512*1:512*2,512*2:512*3] = np.transpose(depth)[::-1,:]

        cv2.imwrite(write_dir1,out)
