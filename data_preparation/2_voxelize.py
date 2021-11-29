import os
import sys
from sort_nicely import *


target_root = sys.argv[1] + "/"
if not os.path.exists(target_root):
    print("ERROR: this dir does not exist: " + target_root)
    exit()

print("2_voxelize.py")

for shape_name in sort_nicely(os.listdir(target_root)):   
    shape_root = target_root + shape_name
    if not os.path.isdir(shape_root):
        continue
    meshes_root = shape_root + "/new_objs/" 

    for obj_name in sort_nicely(os.listdir(meshes_root)):
        if not obj_name.endswith(".obj"):
            continue
        mesh_name = meshes_root + obj_name 
        # print(mesh_name)

        maxx = 0.5
        maxy = 0.5
        maxz = 0.5
        minx = -0.5
        miny = -0.5
        minz = -0.5

        command = "./binvox -bb " + str(minx) + " " + str(miny) + " " + str(minz) + " " + str(maxx) + \
            " " + str(maxy) + " " + str(maxz) + " " + " -d 64 -e " + mesh_name + " >> " + \
            shape_root + "/voxel_conversions.txt"
        os.system(command)

    voxels_root = shape_root + "/voxels/"
    if not os.path.exists(voxels_root):
        os.makedirs(voxels_root)

    for voxel_name in sort_nicely(os.listdir(meshes_root)):
        if voxel_name.endswith(".binvox"):
            os.system("mv " + meshes_root + voxel_name + " " + voxels_root + voxel_name)
