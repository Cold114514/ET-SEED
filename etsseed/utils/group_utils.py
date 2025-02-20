import torch
import numpy as np
import torch.nn.functional as F
from pdb import set_trace as bp

def se2_to_SE3(se2):
    # [batch, x, z, theta] -> [batch, 4, 4]
    # given a 3D vector in se2(translation and rotation on x-z plane), convert it to a 4x4 matrix of SE3
    assert se2.shape[-1] == 3
    batch_size = se2.shape[0]
    se3 = torch.zeros(batch_size, 4, 4, device=se2.device)
    se3[:, 0, 0] = torch.cos(se2[:, 2])
    se3[:, 0, 2] = -torch.sin(se2[:, 2])
    se3[:, 1, 1] = 1
    se3[:, 2, 0] = torch.sin(se2[:, 2])
    se3[:, 2, 2] = torch.cos(se2[:, 2])
    se3[:, 0, 3] = se2[:, 0]
    se3[:, 2, 3] = se2[:, 1]
    se3[:, 3, 3] = 1
    return se3

def SE3_to_se2(SE3):
    # [batch, 4, 4] -> [batch, x, z, theta]
    # given a 4x4 matrix of SE3, convert it to a 3D vector in se2(translation and rotation on x-z plane)
    #!!! The transformation is only valid for x-z plane translation and rotation
    assert SE3.shape[-2:] == (4, 4)
    batch_size = SE3.shape[0]
    se2 = torch.zeros(batch_size, 3, device=SE3.device)
    se2[:, 0] = SE3[:, 0, 3]
    se2[:, 1] = SE3[:, 2, 3]
    se2[:, 2] = torch.atan2(SE3[:, 2, 0], SE3[:, 0, 0])
    return se2

def bgs(d6s):
    # print(d6s.shape)
    b_copy = d6s.clone()
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1),
                                    a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1)

def process_action(raw_action, translation):
    # [batch, 9] -> [batch, 4, 4]
    # given a 9D vector of action(3translation + 6rotation), convert it to a 4x4 matrix of SE3
    # assert raw_action.shape[-1] == 9
    batch_size = raw_action.shape[0]
    action = torch.zeros(batch_size, 4, 4, device=raw_action.device)
    action[:,3,3] = 1
    action[:,:3,3] += translation[:,:] # translation
    R = bgs(raw_action[:,3:].reshape(-1, 2, 3).permute(0, 2, 1))
    action[:,:3,:3] += R # rotation
    return action


def orthogonalization(raw_action):
    # [batch, 9] -> [batch, 4, 4]
    batch_size = raw_action.shape[0]
    R = bgs(raw_action[:,3:].reshape(-1, 2, 3).permute(0, 2, 1))
    return R

def bgdR(Rgts, Rps):
    Rgts = Rgts.float()
    Rps = Rps.float()
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    return torch.acos(theta)