import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from isaacgym import gymapi
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

def transform_to_SE3(transform):
    translation = np.array([transform.p.x, transform.p.y, transform.p.z])
    #! scipy rotation is in the format of (x, y, z, w) (scalar last)
    #! gymapi rotation is also in the format of (x, y, z, w) (scalar last)
    Q = np.array([transform.r.x, transform.r.y, transform.r.z, transform.r.w])
    r = R.from_quat(Q)
    Rot = r.as_matrix()
    SE3 = np.eye(4)
    SE3[:3, :3] = Rot
    SE3[:3, 3] = translation
    return SE3

def SE3_to_transform(SE3):
    translation = SE3[:3, 3]
    r = R.from_matrix(SE3[:3, :3])
    Q = r.as_quat()
    return gymapi.Transform(gymapi.Vec3(translation[0], translation[1], translation[2]), gymapi.Quat(Q[0], Q[1], Q[2], Q[3]))



# SE3 = se2_to_SE3(torch.tensor([[1, 2, np.pi/6]], dtype=torch.float32))
# se2 = SE3_to_se2(SE3)
# print(se2)
