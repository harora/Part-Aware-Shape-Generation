import os
import sys


dataset_path = sys.argv[1]
os.system("python 1_simplify_obj.py " + dataset_path)
os.system("python 2_voxelize.py " + dataset_path)
os.system("python 3_floodfill.py " + dataset_path + " 0 4")
os.system("python 3_floodfill.py " + dataset_path + " 1 4")
os.system("python 3_floodfill.py " + dataset_path + " 2 4")
os.system("python 3_floodfill.py " + dataset_path + " 3 4")
os.system("python 4_validate.py " + dataset_path)
