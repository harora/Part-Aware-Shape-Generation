import numpy as np
import os
import argparse
from glob import glob
import json
import open3d as o3d
from multiprocessing import Pool, cpu_count


parser = argparse.ArgumentParser()
parser.add_argument("-pathIn","--pathIn", type=str)

def lod_mesh_export(mesh, lods):
    mesh_lods = []
    for i, lod_ in enumerate(lods):
        mesh_lod = mesh.simplify_quadric_decimation(lod_)
        mesh_lods.append(mesh_lod)
        print("generation of "+str(lod_)+" LoD successful")
    return mesh_lods

def reconstructPcd(point_cloud, outputdir, fname):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))

    pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.9, max_nn=30))
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    pcd.orient_normals_consistent_tangent_plane(100)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    print(mesh)
    
    fname = fname.replace('.npz', '')
    outputname = os.path.join(outputdir, f"{fname}.obj")
    o3d.io.write_triangle_mesh(outputname, mesh)
    
    return
    # lods = [100000,50000,10000,1000,100]
    # my_lods = lod_mesh_export(mesh, lods)
    # for idx, mesh_ in enumerate(my_lods):
    #     outputname = os.path.join(outputdir, f"{fname}_{lods[idx]}.ply")
    #     o3d.io.write_triangle_mesh(outputname, mesh_)
    # o3d.visualization.draw_geometries([mesh_])

def processPcdFile(pcdname):
    outdir = os.path.dirname(pcdname)
    outdir = outdir + "_obj"
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(pcdname)
    print(f"Reading pcd from {pcdname}")
    points = np.load(pcdname)
    pts = points['xyz']
    max_ = np.max(pts)
    std_ = np.std(pts)
    pts = (pts-std_)/max_
    reconstructPcd(pts, outdir, name)
    nn = os.path.join(outdir, name + ".txt")
    with open(nn, 'w') as f:
        np.savetxt(f, points['xyz'])
if __name__ == "__main__":
    args = parser.parse_args()
    pcdPath = args.pathIn
    # pcdPath = "single_pcd"
    print(f"Processing pcds at {pcdPath}")
    pointclouds = glob(os.path.join(pcdPath, '**', '*.npz'))
    
    cores = cpu_count()
    noProcess = min(cores, len(pointclouds))
    print(f"Cores: {cores}, No processes: {noProcess}")
    if noProcess > 1:
        with Pool(noProcess) as pool:
            results = [pool.map(processPcdFile, (name,)) for name in pointclouds]
        # for res in results:
        #     print(res)
    else:
        [processPcdFile(name) for name in pointclouds]
    print("Done")
        #write to text