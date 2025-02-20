import torch

def calculate_norm_loss(output_directions):
    # calculate ||R^T*R - R_trace||_F
    R = output_directions.reshape(-1, 3, 3)
    R = torch.bmm(R.permute(0, 2, 1), R)    # column-wise
    norm_loss = (torch.bmm(R.permute(0, 2, 1), R) - torch.eye(3).repeat(R.shape[0], 1, 1).to(output_directions.device)).norm()

    return norm_loss

def R_to_phi(R):
    bs = R.shape[0]
    phi = torch.zeros(bs, 3).to(R.device)
    phi[:, 0] = R[:, 2, 1]
    phi[:, 1] = R[:, 0, 2]
    phi[:, 2] = R[:, 1, 0]

    return phi
    
def geodesic_distance_between_R(R1, R2):
    R1_T = R1.transpose(1, 2)
    R = torch.einsum("bmn,bnk->bmk", R1_T, R2)
    diagonals = torch.diagonal(R, dim1=1, dim2=2)
    traces = torch.sum(diagonals, dim=1).unsqueeze(1)   # [bs]
    theta = torch.clamp(0.5 * (traces - 1), -1 + 1e-6, 1 - 1e-6)
    dist = torch.acos(theta)

    return dist

def double_geodesic_distance_between_poses(T1, T2, return_both=False):
    R_1, t_1 = T1[:, :3, :3], T1[:, :3, 3]
    R_2, t_2 = T2[:, :3, :3], T2[:, :3, 3]

    dist_R_square = geodesic_distance_between_R(R_1, R_2) ** 2
    dist_t_square = torch.sum((t_1-t_2) ** 2, dim=1)
    dist = torch.sqrt(dist_R_square.squeeze(-1) + dist_t_square)    # [bs]

    if return_both:
        return torch.sqrt(dist_t_square).mean(), torch.sqrt(dist_R_square).mean()
    else:
        return dist.mean()
    
    
def compute_loss(T1, T2):
    R_1, t_1 = T1[:, :3, :3], T1[:, :3, 3]
    R_2, t_2 = T2[:, :3, :3], T2[:, :3, 3]
    dist_R_square = geodesic_distance_between_R(R_1, R_2) ** 2
    dist_t_square = torch.sum((t_1-t_2) ** 2, dim=1)
    # dist = torch.sqrt(dist_R_square.squeeze(-1) + dist_t_square)    # [bs]
    dist_R = torch.sqrt(dist_R_square).mean()
    dist_T = torch.sqrt(dist_t_square).mean()
    dist = dist_R + dist_T
    return dist, dist_R, dist_T