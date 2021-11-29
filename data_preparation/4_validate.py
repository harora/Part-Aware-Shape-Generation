import os
import sys
from sort_nicely import *


source_root = sys.argv[1] + "/"
if not os.path.exists(source_root):
    print("ERROR: this dir does not exist: " + source_root)
    exit()

print("4_validate.py")

for shape_name in sort_nicely(os.listdir(source_root)):   
    shape_root = source_root + shape_name
    if not os.path.isdir(shape_root):
        continue

    assert(os.path.exists(shape_root + "/new_objs/")) 
    assert(os.path.exists(shape_root + "/voxels/")) 
    assert(os.path.exists(shape_root + "/filled_voxels/"))


        