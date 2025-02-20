import torch
import numpy as np
import torch.nn.functional as F
from pdb import set_trace as bp


def depth_image_to_point_cloud(rgb, depth, scale, K, pose):
    ##!!! chenrui: This implementation is wrong, just work for the toy environment
    # print(rgb.shape, depth.shape)
    # print(K)
    # print(pose)
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])
    # print(depth)
    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)
    # Z = depth.astype(float) / scale
    Z = depth
    np.savetxt('depth.txt',Z)
    Z = np.ravel(Z)
    valid = (Z>-175)
    # X = (u - K[0, 2]) * Z / K[0, 0]
    # Y = (v - K[1, 2]) * Z / K[1, 1]

    X = u
    Y = v

    X = np.ravel(X)
    Y = np.ravel(Y)
    

    
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z))
    # position = np.vstack((X, Y, Z, np.ones(len(X))))
    # position = np.dot(pose, position)
    # position /= position[3, :]
    # position = position[0:3, :].T 

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3,:], R, G, B)))
    return points