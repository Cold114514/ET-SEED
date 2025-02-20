import open3d as o3d
import torch
import numpy as np
import colorsys

def interpolate_color(weight, color1=[0, 0, 0.4], color2=[1, 0.1, 0]):
    return [c1 * (1 - weight) + c2 * weight for c1, c2 in zip(color1, color2)]

def save_pcd_as_pcd(xyz, rgb, save_file="./pcd/test.pcd", draw_heatmap=False):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
        rgb = rgb.detach().cpu().numpy()

    pcd=o3d.geometry.PointCloud()

    if draw_heatmap:
        rgb = np.array([interpolate_color(hm) for hm in rgb.mean(axis=-1)])

    pcd.points=o3d.utility.Vector3dVector(xyz)
    pcd.colors=o3d.utility.Vector3dVector(rgb)

    o3d.io.write_point_cloud(save_file, pcd)
