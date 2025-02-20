import torch
import torch_scatter
import numpy as np
import dgl
from dgl import backend as F


from dgl.convert import graph as dgl_graph
from networks.se3_transformer.model.fiber import Fiber
from torch_cluster import radius_graph
from scipy.spatial.transform import Rotation

def voxel_filter(pcd, feature, voxel_size: float, coord_reduction: str = "average"):
    device = pcd.device

    mins = pcd.min(dim=-2).values

    vox_idx = torch.div((pcd - mins), voxel_size, rounding_mode='trunc').type(torch.long)
    shape = vox_idx.max(dim=-2).values + 1
    raveled_idx = torch.tensor(np.ravel_multi_index(vox_idx.T.cpu().numpy(), shape.cpu().numpy()), device = device, dtype=vox_idx.dtype)

    n_pts_per_vox = torch_scatter.scatter(torch.ones_like(raveled_idx, device=device), raveled_idx, dim_size=shape[0]*shape[1]*shape[2])
    nonzero_vox = n_pts_per_vox.nonzero()
    n_pts_per_vox = n_pts_per_vox[nonzero_vox].squeeze(-1)
    
    color_vox = torch_scatter.scatter(feature, raveled_idx.unsqueeze(-1), dim=-2, dim_size=shape[0]*shape[1]*shape[2])[nonzero_vox].squeeze(-2)
    color_vox /= n_pts_per_vox.unsqueeze(-1)

    if coord_reduction == "center":
        coord_vox = np.stack(np.unravel_index(nonzero_vox.cpu().numpy().reshape(-1), shape.cpu().numpy()), axis=-1)
        coord_vox = torch.tensor(coord_vox, device = device, dtype=vox_idx.dtype)
        coord_vox = coord_vox * voxel_size + mins + (voxel_size/2)
    elif coord_reduction == "average":
        coord_vox = torch_scatter.scatter(pcd, raveled_idx.unsqueeze(-1), dim=-2, dim_size=shape[0]*shape[1]*shape[2])[nonzero_vox].squeeze(-2)
        coord_vox /= n_pts_per_vox.unsqueeze(-1)
    else:
        raise ValueError(f"Unknown coordinate reduction method: {coord_reduction}")

    return coord_vox, color_vox


def build_graph(xyz, feature, dist_threshold=0.02, self_connection=True, voxelize=False, voxel_size=0.001, fiber_in=Fiber({0: 3})):
    # xyz can be the same shape or not    
    if isinstance(xyz, torch.Tensor):
        bs = xyz.shape[0]
    else:
        bs = len(xyz)

    batched_graph = []
    pcds = []
    raw_features = []

    max_num_neighbors = 512

    if self_connection is True:
        max_num_neighbors += 1
    
    for pcd_index in range(bs):
        current_pcd, current_feature = xyz[pcd_index], feature[pcd_index]

        if voxelize:
            current_pcd, current_feature = voxel_filter(current_pcd, current_feature, voxel_size=voxel_size)

        edge_src, edge_dst = radius_graph(current_pcd, dist_threshold * 0.999, max_num_neighbors = max_num_neighbors, loop = self_connection)

        pcds.append(current_pcd)
        raw_features.append(current_feature)
        g = dgl_graph((edge_src, edge_dst))

        g = to_cuda(g)
        g.ndata["pos"] = F.tensor(current_pcd, dtype=F.data_type_dict["float32"])
        g.ndata["attr"] = F.tensor(current_feature, dtype=F.data_type_dict["float32"])
        g.edata["rel_pos"] = F.tensor(current_pcd[edge_dst] - current_pcd[edge_src])
        g.edata['edge_attr'] = g.edata["rel_pos"].norm(dim=-1)

        batched_graph.append(g)

    batched_graph = dgl.batch(batched_graph)
    batched_graph = to_cuda(batched_graph)

    # node features
    node_feats = {}
    degrees = fiber_in.degrees
    channels = fiber_in.channels
    start = 0
    for i in range(len(degrees)):
        feat = batched_graph.ndata['attr'][:, start:start+channels[i]*((2*degrees[i])+1)]
        node_feats[str(degrees[i])] = feat.reshape(feat.shape[0], channels[i], 2*degrees[i]+1)  # n, channel, 2l+1
        start += channels[i]*((2*degrees[i])+1)
    
    edge_feats = {'0': batched_graph.edata['edge_attr'].unsqueeze(-1).unsqueeze(-1)}

    return batched_graph, node_feats, edge_feats, pcds, raw_features

def modified_gram_schmidt(tensor, to_cuda=False):
    rows, cols = tensor.shape
    
    ortho_tensor = torch.empty_like(tensor)
    if to_cuda:
        ortho_tensor = ortho_tensor.cuda()
    
    for i in range(cols):
        v = tensor[:, i]
        
        for j in range(i):
            u = ortho_tensor[:, j]
            projection = torch.dot(v, u) / torch.dot(u, u)
            v = v - projection * u

        for j in range(i):
            u = ortho_tensor[:, j]
            v = v - torch.dot(v, u) * u
        
        ortho_tensor[:, i] = v / torch.norm(v)
    
    return ortho_tensor


def to_cuda(x):
    """ Try to convert a Tensor, a collection of Tensors or a DGLGraph to CUDA """
    if isinstance(x, Tensor):
        return x.cuda(non_blocking=True)
    elif isinstance(x, tuple):
        return (to_cuda(v) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=torch.cuda.current_device(), non_blocking=True)
    
    


