import torch
import numpy as np
import torch.nn.functional as F
from pdb import set_trace as bp
import pytorch3d.ops as torch3d_ops

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0).cpu()
    else:
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
    return sampled_points, indices


def depth_image_to_point_cloud(task_name, rgb, depth, scale, vinv, proj, use_cuda = True, index =0):
    assert task_name in ['door_opening', 'rotate_triangle']
    H, W = depth.shape
    depth[depth == -np.inf] = 0
     # clamp depth image to 10 meters to make output image human friendly
    depth[depth < -10] = -10
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])
    u, v = np.meshgrid(u, v)
    u = torch.tensor(u)
    v = torch.tensor(v)

    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]
    centerU = W / 2
    centerV = H / 2
    Z = depth

    X = -(u - centerU) / W * Z[v, u] * fu
    Y = (v - centerV) / H * Z[v, u] * fv
    X = X.view(-1)
    Y = Y.view(-1)
    Z = Z.view(-1)

    rgb = rgb.view(-1, 4)
    # print(torch.max(Z), torch.min(Z), torch.mean(Z))
    if task_name == 'rotate_triangle':
        valid = (rgb[..., 0] > 250) & (rgb[..., 0] > rgb[..., 1]) & (rgb[..., 0] > rgb[..., 2])
    elif task_name == 'door_opening':
        valid = ((Z > -1))
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    position = torch.vstack((X, Y, Z, torch.ones(len(X))))
    position = position.permute(1, 0)
    position = position @ vinv
    points = position[:, 0:3]

    valid2 = points[:, 1] > 0.02
    # print('valid', torch.sum(valid))
    # print('valid2', torch.sum(valid2))
    points = points[valid2]

    RGB = rgb.view(-1, 4)[valid][valid2]
    R = RGB[:, 0].unsqueeze(1) # [num_points,1]
    G = RGB[:, 1].unsqueeze(1)
    B = RGB[:, 2].unsqueeze(1)

    #for visualization
    # with open(f'log/points{index}.obj', 'w') as f:
    #     for i in range(points.shape[0]):
    #         f.write('v {} {} {} {} {} {}\n'.format(points[i, 0], points[i, 1], points[i, 2], float(R[i]/255), float(G[i]/255), float(B[i]/255)))
    
    # points: [num_points,3]
    # sample_indices: [1, num_points]
    if task_name == 'door_opening':
        num_points = 1024
    elif task_name == 'rotate_triangle':
        num_points = 100
    points, sample_indices = farthest_point_sampling(points, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    # print('sample_indices',sample_indices.shape)
    R = torch.index_select(R, dim=0, index=sample_indices.squeeze(0))
    G = torch.index_select(G, dim=0, index=sample_indices.squeeze(0))
    B = torch.index_select(B, dim=0, index=sample_indices.squeeze(0))    
    points = torch.cat((points, R, G, B), 1)
    return points


def depth_image_to_point_cloud_painting(rgb, depth, scale, vinv, proj, use_cuda = True, index =0):

    H, W = depth.shape
    depth[depth == -np.inf] = 0
     # clamp depth image to 10 meters to make output image human friendly
    depth[depth < -10] = -10
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])
    u, v = np.meshgrid(u, v)
    u = torch.tensor(u)
    v = torch.tensor(v)

    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]
    centerU = W / 2
    centerV = H / 2
    Z = depth

    X = -(u - centerU) / W * Z[v, u] * fu
    Y = (v - centerV) / H * Z[v, u] * fv
    X = X.view(-1)
    Y = Y.view(-1)
    Z = Z.view(-1)

    rgb = rgb.view(-1, 4)
    # print(torch.max(Z), torch.min(Z), torch.mean(Z))
    valid = (Z > -0.5) & (rgb[..., 0] > 250)
    # print(torch.sum(valid))
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    # print(X.shape, Y.shape, Z.shape)
    position = torch.vstack((X, Y, Z, torch.ones(len(X))))
    position = position.permute(1, 0)
    position = position @ vinv
    points = position[:, 0:3]

    RGB = rgb.view(-1, 4)[valid]
    R = RGB[:, 0].unsqueeze(1) # [num_points,1]
    G = RGB[:, 1].unsqueeze(1)
    B = RGB[:, 2].unsqueeze(1)

    #for visualization
    # with open(f'log/points{index}.obj', 'w') as f:
    #     for i in range(points.shape[0]):
    #         f.write('v {} {} {} {} {} {}\n'.format(points[i, 0], points[i, 1], points[i, 2], float(R[i]/255), float(G[i]/255), float(B[i]/255)))
    
    # points: [num_points,3]
    # sample_indices: [1, num_points]
    num_points = 256
    points, sample_indices = farthest_point_sampling(points, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    # print('sample_indices',sample_indices.shape)
    R = torch.index_select(R, dim=0, index=sample_indices.squeeze(0))
    G = torch.index_select(G, dim=0, index=sample_indices.squeeze(0))
    B = torch.index_select(B, dim=0, index=sample_indices.squeeze(0))
    points = torch.cat((points, R, G, B), 1)
    return points