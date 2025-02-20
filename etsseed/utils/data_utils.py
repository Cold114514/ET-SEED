import torch
from torch.utils.data import Dataset
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.axangles import axangle2mat, mat2axangle
from copy import deepcopy
from math import factorial

class SE3Demo(Dataset):
    def __init__(self, demo_dir, data_aug=False, device="cuda:0", aug_methods=[]):
        super(SE3Demo, self).__init__()

        self.device = device
        demo = np.load(demo_dir, allow_pickle=True) 
        if isinstance(demo, np.ndarray):
            demo = demo.item()
        traj_num, video_len, point_num, _ = demo["xyz"].shape

        self.xyz = torch.from_numpy(demo["xyz"][:, 0:20, ...]).float().to(self.device).reshape(-1, point_num, 3)
        self.rgb = torch.from_numpy(demo["rgb"][:, 0:20, ...]).float().to(self.device).reshape(-1, point_num, 3)
        self.seg_center = torch.from_numpy(demo["seg_center"][:, 0:20, ...]).float().to(self.device).reshape(-1, 3)
        self.axes = torch.from_numpy(demo["axes"][:, 0:20, ...]).float().to(self.device).reshape(-1, 9)

        self.data_aug = data_aug
        self.aug_methods = aug_methods

    def __len__(self):
        return self.xyz.shape[0]

    def __getitem__(self, index):
        data = {
            "xyz": self.xyz[index],
            "rgb": self.rgb[index],
            "seg_center": self.seg_center[index],
            "axes": self.axes[index],
        }

        with torch.no_grad():
            if self.data_aug:
                for method in self.aug_methods:
                    data = globals()[method](deepcopy(data))

        return data

def seg_pointcould_with_boundary(xyz, rgb, xyz_for_seg, x_min, x_max, y_min, y_max, z_min, z_max):
    rgb = rgb[
        (xyz_for_seg[:, 0] >= x_min) & (xyz_for_seg[:, 0] <= x_max) &
        (xyz_for_seg[:, 1] >= y_min) & (xyz_for_seg[:, 1] <= y_max) &
        (xyz_for_seg[:, 2] >= z_min) & (xyz_for_seg[:, 2] <= z_max)
    ]
    xyz = xyz[
        (xyz_for_seg[:, 0] >= x_min) & (xyz_for_seg[:, 0] <= x_max) &
        (xyz_for_seg[:, 1] >= y_min) & (xyz_for_seg[:, 1] <= y_max) &
        (xyz_for_seg[:, 2] >= z_min) & (xyz_for_seg[:, 2] <= z_max)
    ]
    return {
        "xyz": xyz,
        "rgb": rgb
    }

def seg_pointcloud(xyz, rgb, reference_point, distance=0.3, extra_data=None):
    distances = torch.norm(xyz - reference_point, dim=1)
    xyz = xyz[distances < distance]
    rgb = rgb[distances < distance]

    data = {
        "xyz": xyz,
        "rgb": rgb,
    }

    if extra_data != None:
        for k in list(extra_data.keys()):
            if k == "xyz" or k == "rgb":
                continue
            data[k] = extra_data[k][distances < distance]
            if torch.isnan(data[k]).any():
                print("Nan detected!")
    return data


def dropout_to_certain_number(xyz, rgb, target_num):
    current_num = xyz.shape[0]
    assert target_num < current_num
    random_indices = torch.randperm(xyz.shape[0])[:target_num]

    xyz = xyz[random_indices]
    rgb = rgb[random_indices]

    return {
        "xyz": xyz,
        "rgb": rgb,
    }


def random_dropout(xyz, rgb, remain_point_ratio=[0.5, 0.9]):
    desired_points = int(np.random.uniform(remain_point_ratio[0], remain_point_ratio[1]) * xyz.shape[0])
    random_indices = torch.randperm(xyz.shape[0])[:desired_points]

    xyz = xyz[random_indices]
    rgb = rgb[random_indices]

    return {
        "xyz": xyz,
        "rgb": rgb,
    }

def downsample_table(data, reference_rgb=[0.1176, 0.4392, 0.4078], total_points=2500):
    xyz = data["xyz"]
    rgb = data["rgb"]
    table_mask = (rgb - torch.tensor(reference_rgb).to(xyz.device)).norm(dim=1) < 0.1
    table_xyz = xyz[table_mask]
    table_rgb = rgb[table_mask]

    non_table_xyz = xyz[~table_mask]
    non_table_rgb = rgb[~table_mask]

    if non_table_xyz.shape[0] <= total_points:
        desired_points = total_points - non_table_xyz.shape[0]
        random_indices = torch.randperm(table_xyz.shape[0])[:desired_points]

        xyz = torch.cat([table_xyz[random_indices], non_table_xyz], dim=0)
        rgb = torch.cat([table_rgb[random_indices], non_table_rgb], dim=0)
    else:   # all from objects
        random_indices = torch.randperm(non_table_xyz.shape[0])[:total_points]

        xyz = non_table_xyz[random_indices]
        rgb = non_table_rgb[random_indices]

    data["xyz"] = xyz
    data["rgb"] = rgb
    return data

def jitter(data, std=0.03):
    data["xyz"] = data["xyz"] + std * torch.randn(data["xyz"].shape).to(data["xyz"].device)
    return data

def random_dropping_color(data, drop_ratio=0.3):
    # randomly remove some points' RGB to [0,0,0]
    N = data["xyz"].shape[0]
    mask = np.random.choice([0, 1], size=N, replace=True, p=[1-drop_ratio, drop_ratio])
    data["rgb"][np.where(mask)] = torch.tensor([0., 0., 0.]).to(data["rgb"].device)
    
    return data

def color_jitter(data, std=0.005):
    data["rgb"] = torch.clamp(data["rgb"] + (torch.rand(data["rgb"].shape).to(data["xyz"].device) - 0.5) * 2 * std, 0, 1)
    return data

def zero_color(data):
    # remove all color
    data["rgb"] = torch.zeros_like(data["rgb"])
    return data

from torchvision.transforms import functional as F
def hsv_transform(data, hue_shift_range=[-0.4, 0.4], sat_shift_range=[0.5, 1.5], val_shift_range=[0.5, 2]):
    img_rgb = data["rgb"].T.unsqueeze(-1) # [N, 3] -> [3, N] -> [3, N, 1], and the adjust functions requires [3, H, W]

    hue_shift = np.random.random_sample() * (hue_shift_range[1] - hue_shift_range[0]) + hue_shift_range[0]
    sat_shift = np.random.random_sample() * (sat_shift_range[1] - sat_shift_range[0]) + sat_shift_range[0]
    val_shift = np.random.random_sample() * (val_shift_range[1] - val_shift_range[0]) + val_shift_range[0]

    img_rgb = F.adjust_hue(img_rgb, hue_factor=hue_shift)
    img_rgb = F.adjust_saturation(img_rgb, saturation_factor=sat_shift)
    img_rgb = F.adjust_brightness(img_rgb, brightness_factor=val_shift)

    data["rgb"] = img_rgb.squeeze(-1).T

    return data

import potpourri3d as pp3d
def geodesic_distance_from_pcd(point_cloud, keypoint_index):
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    solver = pp3d.PointCloudHeatSolver(point_cloud)

    # Compute the geodesic distance to point 4
    dists = solver.compute_distance(keypoint_index)

    return torch.from_numpy(dists).float()

def get_heatmap(point_cloud, keypoint_index, distance="geodesic", max_value = 10.0, std_dev=0.005):
    # distance: "l2" or "geodesic"

    # Extract keypoint coordinates
    keypoint = point_cloud[keypoint_index]
    if distance == "l2":
        # Compute the L2 distance from the keypoint to all other points
        distances = torch.norm(point_cloud - keypoint, dim=1)
    elif distance == "geodesic":
        # Compute the geodesic distance from the keypoint to all other points
        distances = geodesic_distance_from_pcd(point_cloud, keypoint_index)
    heatmap_values = torch.exp(-0.5 * (distances / std_dev) ** 2)
    heatmap_values /= torch.max(heatmap_values)

    heatmap_values *= max_value

    return heatmap_values

import random

def mask_part_point_cloud(xyz, rgb, mask_radius=0.015):
    N, _ = xyz.shape

    center_idx = random.randint(0, N-1)
    center_point = xyz[center_idx]

    distances = torch.sqrt(torch.sum((xyz - center_point) ** 2, dim=1))
    mask = distances < mask_radius

    masked_point_cloud = xyz[~mask]
    masked_rgb = rgb[~mask]

    return {
        "xyz": masked_point_cloud,
        "rgb": masked_rgb,
    }

def bezier_curve(points, x_coord):
    def getB(i):
        t = np.math.factorial(n)*init_t**i*(1-init_t)**(n-i)/(np.math.factorial(i)*np.math.factorial(n-i))
        return np.array([t,t]).T
    n = points.shape[0]-1
    init_t = np.linspace(0,1,1000)
    P = np.zeros((1000,2))
    for i in range(n+1):
        P += getB(i)*points[i]
    y_coord = np.interp(x_coord, P[:,0], P[:,1])
    return y_coord
