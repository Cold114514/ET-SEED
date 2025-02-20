import torch
import os
from etseed.model.se3_transformer.se3_backbone import SE3Backbone, ExtendedModule
from etseed.model.se3_transformer.se3_transformer.model.fiber import Fiber
from etseed.utils.data_utils import seg_pointcloud, random_dropout, mask_part_point_cloud
from etseed.utils.group_utils import process_action, orthogonalization
from etseed.utils.vis import save_pcd_as_pcd
from etseed.utils.data_utils import get_heatmap
from pdb import set_trace as bp

class SE3ManiNet_Equivariant(ExtendedModule):
    def __init__(self, voxelize=True, voxel_size=0.01, radius_threshold=0.12, feature_point_radius=0.02):
        super().__init__()
        ''' 
        input features:
        rgb: 3 type0
        action: 9 type0 (the first 2 columns of rotation matrix(2x3), and a translation vector(1x3))
        pose: 9 type0 (the first 2 columns of rotation matrix(2x3), and a translation vector(1x3))
        k: 1 type0 (denote currently is the k-th denoising step)
        
        output features:
        action: 2 type1(rotation) + 3 type0(translation)
        '''
        num_fib_in = [13]
        num_fib_out = [3,2]
        self.pos_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
            }),
            fiber_out=Fiber({
                "0": 4, # 1: heatmap, 3: offset
                "1": 2  # 2: ori
            }),
            num_layers= 4,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize, # change this to control num_points
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )
        self.ori_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
            }),
            fiber_out=Fiber({
                "0": num_fib_out[0], # 3 type 0
                "1": num_fib_out[1], # 2 type 1
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )
        self.feature_point_radius = feature_point_radius

    def forward(self, inputs, train_pos=False, reference_point=None, distance_threshold=0.3, random_drop=False, draw_pcd=False, pcd_name=None, mask_part=False, save_ori_feature=False):
        bs = inputs["xyz"].shape[0]
        seg_output = self.pos_net(inputs)
        xyz = seg_output["xyz"]
        feature = seg_output["feature"] # encode node feature
        output_pos = torch.zeros([len(xyz), 3]).to(self.device)
        mass_center = torch.zeros([len(xyz), 3]).to(self.device)
        
        action_raw=[]
        for i in range(bs):
            action_raw.append(torch.mean(seg_output["feature"][i][:,1:], dim=0))  # 9
        action_raw = torch.stack(action_raw, dim=0)  # batch_size * 9
        R = orthogonalization(action_raw)
        
        for i in range(len(xyz)):
            pos_weight = torch.nn.functional.softmax(feature[i][:,:1].reshape(-1, 1), dim=0).squeeze()
            offset = torch.mean(feature[i][:,1:4], dim=0)
            mass_center[i] = (xyz[i].T * pos_weight).T.sum(dim=0)
            offset = offset @ R[i]
            output_pos[i] = mass_center[i] + offset
            
        action = process_action(action_raw, output_pos)
        return action

class SE3ManiNet_Equivariant_Separate(ExtendedModule):
    def __init__(self, voxelize=True, voxel_size=0.01, radius_threshold=0.12, feature_point_radius=0.02):
        super().__init__()
        ''' 
        input features:
        rgb: 3 type0
        action: 9 type0 (the first 2 columns of rotation matrix(2x3), and a translation vector(1x3))
        pose: 9 type0 (the first 2 columns of rotation matrix(2x3), and a translation vector(1x3))
        k: 1 type0 (denote currently is the k-th denoising step)

        output features:
        action: 2 type1(rotation) + 3 type0(translation)
        '''
        num_fib_in = [13]
        num_fib_out = [3,2]
        self.pos_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
            }),
            fiber_out=Fiber({
                "0": 4, # 1: heatmap, 3: offset
                "1": 2  # 2: ori
            }),
            num_layers= 4,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize, # change this to control num_points
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )
        self.ori_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
            }),
            fiber_out=Fiber({
                "0": num_fib_out[0], # 4 type 0
                "1": num_fib_out[1], # 2 type 1
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )
        self.feature_point_radius = feature_point_radius

    def forward(self, inputs, train_pos=False, reference_point=None, distance_threshold=0.3, random_drop=False, draw_pcd=False, pcd_name=None, mask_part=False, save_ori_feature=False):
        bs = inputs["xyz"].shape[0]
        seg_output = self.pos_net(inputs)
        xyz = seg_output["xyz"]
        feature = seg_output["feature"] # encode node feature
        output_pos = torch.zeros([len(xyz), 3]).to(self.device)
        mass_center = torch.zeros([len(xyz), 3]).to(self.device)
        
        action_raw=[]
        for i in range(bs):
            action_raw.append(torch.mean(seg_output["feature"][i][:,1:], dim=0))  # 9
        action_raw = torch.stack(action_raw, dim=0)  # batch_size * 9
        R = orthogonalization(action_raw)
        
        for i in range(len(xyz)):
            pos_weight = torch.nn.functional.softmax(feature[i][:,:1].reshape(-1, 1), dim=0).squeeze()
            offset = torch.mean(feature[i][:,1:4], dim=0)
            mass_center[i] = (xyz[i].T * pos_weight).T.sum(dim=0)
            offset = offset @ R[i]
            output_pos[i] = mass_center[i] + offset
        
        ori_output = self.ori_net(inputs)
        action_ori = []
        for i in range(bs):
            action_ori.append(torch.mean(ori_output["feature"][i], dim=0))  # 9
        action_ori = torch.stack(action_ori, dim=0)  # batch_size * 9
        
        action = process_action(action_ori, output_pos)
        return action       


class SE3ManiNet_Invariant(ExtendedModule):
    def __init__(self, voxelize=True, voxel_size=0.01, radius_threshold=0.12, feature_point_radius=0.02, T_p=4):
        super().__init__()
        num_fib_in = [13]
        num_fib_out = [9]
        self.pos_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
            }),
            fiber_out=Fiber({
                # "0": 3,
                "1": 1, 
            }),
            num_layers= 4,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )
        self.ori_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
            }),
            fiber_out=Fiber({
                # "0": num_fib_out[0],
                "1": 3
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )
        self.feature_point_radius = feature_point_radius

    def forward(self, inputs, train_pos=False, reference_point=None, distance_threshold=0.3, random_drop=False, draw_pcd=False, pcd_name=None, mask_part=False, save_ori_feature=False):
        bs = inputs["xyz"].shape[0] 
        # process translation
        seg_output = self.pos_net(inputs)
        feature = seg_output["feature"]
        
        feature_list = list()
        for i in range(bs):
            typei_feature = seg_output["feature"][i] # [points, 3]
            actioni_raw = torch.mean(typei_feature,dim = 0)
            feature_list.append(actioni_raw)
        output_pos = torch.stack(feature_list,dim = 0)

        # process orientation
        ori_output = self.ori_net(inputs)
        xyz = ori_output["xyz"]
        feature_list = list()
        for i in range(bs):
            typei_feature = ori_output["feature"][i] # [points, 9]
            actioni_raw = torch.mean(typei_feature,dim = 0)
            feature_list.append(actioni_raw)
        action = torch.stack(feature_list,dim = 0)
        action = process_action(action, output_pos)
        return action # [bz, 4, 4]

    