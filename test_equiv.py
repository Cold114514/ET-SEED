import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from etseed.model.se3_transformer.equinet import SE3ManiNet_Invariant, SE3ManiNet_Equivariant_Separate
from etseed.utils.group_utils import bgs, bgdR
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import tqdm
from pdb import set_trace as bp
import math

#! Check invariant model
def check_invariant_model(num_trials=10, threshold=0.01):
    success_record = []
    for test_inv_trial in tqdm.tqdm(range(num_trials)):
        rot = R.random()
        pts = np.random.rand(100,3)

        rotated_pts = rot.apply(pts) + np.random.rand(3)
        xyz = np.stack([pts, rotated_pts], axis=0)

        feature = np.random.rand(100,13)
        feature = np.stack([feature, feature], axis=0)

        xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
        feature = torch.tensor(feature, dtype=torch.float32).cuda()
        data = {}
        
        data['xyz'] = xyz
        data['feature'] = feature
        model = SE3ManiNet_Invariant().cuda()
        result = model(data)
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())
        result_np = np.array(result_np)
        
        difference = result_np[0] - result_np[1]
        print("difference",difference)
        
        if (np.allclose(result_np[0], result_np[1], atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)
    print('Inv test pass rate:', np.mean(success_record)*100, '%')
    return np.mean(success_record)

#! Check equivariant model
#! Although all the rotations written in the code are right multiplications, the actual physical meaning is left multiplication.
def check_equivariant_model(num_trials=10, threshold=np.pi/180*1):
    model = SE3ManiNet_Equivariant_Separate().cuda()
    total_trials = num_trials
    success_count = 0
    failure_count = 0
    zero_count = 0
    
    total_trans_dist = 0
    total_rot_dist = 0
    total_real_dist = 0
    
    for trial in tqdm.tqdm(range(total_trials)):
        rot = R.random()
        pts = np.random.rand(128,3)
        num_points = pts.shape[0]
        trans = np.random.rand(3) * 10
        T = np.eye(4)
        T[:3,:3] = rot.as_matrix()
        T[:3,3] = trans
        rotated_pts = pts @ rot.as_matrix().T + trans
        xyz = np.stack([pts, rotated_pts], axis=0)
        feature = np.random.rand(num_points,13)
        feature = np.stack([feature, feature], axis=0)
        xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
        feature = torch.tensor(feature, dtype=torch.float32).cuda()
        data = {}
        data['xyz'] = xyz
        data['feature'] = feature
        #! model output
        result_np = []
        result = model(data)
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())
        # phi(Rx)
        phi_Tx = result_np[1]
        # Rphi(x)
        phi_x = result_np[0]
        Tphi_x = np.eye(4)
        Tphi_x[:3,:3] = phi_x[:3,:3] @ rot.as_matrix().T
        Tphi_x[:3,3] = rot.as_matrix() @ phi_x[:3,3] + trans
        if (np.allclose(phi_Tx, np.zeros_like(phi_Tx), atol=1e-1)):
            zero_count += 1
            # print('zero')
            continue
        geo_dist = bgdR(torch.tensor(phi_Tx)[:3,:3].unsqueeze(0), torch.tensor(Tphi_x[:3,:3]).unsqueeze(0)).item()
        trans_dist = np.linalg.norm(phi_Tx[:3,3] - Tphi_x[:3,3])
        
        real_dist = math.sqrt(geo_dist ** 2 + trans_dist ** 2)
        
        total_rot_dist = total_rot_dist + geo_dist
        total_trans_dist = total_trans_dist + trans_dist
        total_real_dist = total_real_dist + real_dist
        print(f"real_dist: {real_dist}")
        if (geo_dist<threshold and trans_dist<threshold):
            success_count += 1
        else:
            failure_count += 1
    print('Equiv test pass rate:', int(success_count/total_trials*100), '%')
    print("geo_dist:",total_rot_dist/total_trials,"trans_dist:",total_trans_dist/total_trials,"real_dist:",total_real_dist/total_trials)

check_invariant_model(num_trials=100, threshold=0.01)
check_equivariant_model(num_trials=100)
#!! test Schimidt
# a = torch.tensor(np.random.rand(10000,3,2)).cuda()
# Sch_a = bgs(a).detach().cpu().numpy()
# for i in range(10000):
#     # print(np.linalg.det(Sch_a[i]))
#     if(not np.allclose(np.linalg.det(Sch_a[i]), 1.0)):
#         print('wrong')
#         bp()