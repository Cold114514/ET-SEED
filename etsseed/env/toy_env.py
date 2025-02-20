"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

from __future__ import print_function, division, absolute_import

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import math
import importlib
import numpy as np
from PIL import Image as im
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import myutils
from copy import copy
from pdb import set_trace as bp
from gym import spaces
import random
import torch
import time

colors = [gymapi.Vec3(1.0, 0.0, 0.0),
          gymapi.Vec3(1.0, 127.0/255.0, 0.0),
          gymapi.Vec3(1.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 0.0, 1.0),
          gymapi.Vec3(39.0/255.0, 0.0, 51.0/255.0),
          gymapi.Vec3(139.0/255.0, 0.0, 1.0)]


class ToyEnv2D():
    def __init__(self, cfg, task_name="rotate_triangle", num_points=1024, save_path="log", save_images=False, save_pts=False):
        
        self.task_name = task_name
        self.cfg = cfg
        self.env = self.create_env(cfg)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.DoF = 3
        self.steps = 0
        self.save_path = save_path + "/" + task_name + "/" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.save_path)
        self.save_images = save_images
        self.save_pts = save_pts

    def get_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        # rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor)
        se2 = self.dof_states[:, 0].view(-1, 3)
        SE3 = myutils.se2_to_SE3(se2)
        return se2, SE3

    def create_env(self, cfg):
        # initialize gym
        self.gym = gymapi.acquire_gym()
        sim_type = cfg.simulator.physics_engine
        sim_params = gymapi.SimParams()
        sim_type = gymapi.SIM_PHYSX
        sim_params.substeps = 2
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 25
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = cfg.simulator.num_threads
        sim_params.physx.use_gpu = cfg.simulator.use_gpu
        sim_params.physx.rest_offset = 0.001
        self.sim = self.gym.create_sim(cfg.simulator.compute_device_id, cfg.simulator.graphics_device_id, sim_type, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # add ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)

        # create viewer
        if cfg.simulator.headless == False:
            viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            l_color = gymapi.Vec3(0.2,0.2,0.2)
            l_ambient = gymapi.Vec3(0.5,0.5,0.5)
            l_direction = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
            if viewer is None:
                print("*** Failed to create viewer")
                quit()
            self.viewer = viewer
        # load assets
        asset_root = "assets"
        table_dims = gymapi.Vec3(0.6, 0.4, 1.0)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002

        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.7, 0.5 * table_dims.y + 0.001, 0.0)


        object_pose = gymapi.Transform()
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        asset_options.fix_base_link = True
        triangle_asset_file = "urdf/EquivDP/triangle.urdf"

        object_assets = []
        object_assets.append(self.gym.load_asset(self.sim, asset_root, triangle_asset_file, asset_options))

        spawn_height = gymapi.Vec3(0.0, 0.1, 0.0)

        corner = table_pose.p - table_dims * 0.5

        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.object_handles = []
        # create env
        env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", 0, 0)
        self.gym.set_rigid_body_color(env,table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))
        x = corner.x + table_dims.x * 0.5# + np.random.rand() * 0.35 - 0.2
        y = table_dims.y 
        z = corner.z + table_dims.z * 0.5# + np.random.rand() * 0.3 - 0.15

        object_pose.p = gymapi.Vec3(x, y, z) + spawn_height
        # print(object_pose.p)
        object_asset = object_assets[0]
        triangle = self.gym.create_actor(env, object_asset, object_pose, "object" , 0, 0)
        self.object_handles.append(triangle)
        #! set object properties
        shape_props = self.gym.get_actor_rigid_shape_properties(env, triangle)
        shape_props[0].friction = 0.
        shape_props[0].rolling_friction = 0.
        shape_props[0].torsion_friction = 0.
        self.gym.set_actor_rigid_shape_properties(env, triangle, shape_props)
        self.gym.set_rigid_body_color(env, self.object_handles[-1], 3, gymapi.MESH_VISUAL_AND_COLLISION, colors[0])


        #! set controller
        props = self.gym.get_actor_dof_properties(env, triangle)
        props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
        props["stiffness"] = (5000.0, 5000.0, 5000.0)
        props["damping"] = (100.0, 100.0, 100.0)
        # print(props)
        self.gym.set_actor_dof_properties(env, triangle, props)
        
        self.camera_handles = []
        #! set camera
        for cam in range(cfg.simulator.cameras):
            camera_props = gymapi.CameraProperties()
            camera_props.width = cfg.simulator.camera_width[cam]
            camera_props.height = cfg.simulator.camera_height[cam]
            camera_position = cfg.simulator.camera_position[cam]
            camera_target = cfg.simulator.camera_target[cam]
            self.camera_handles.append(self.gym.create_camera_sensor(env, camera_props))
            self.gym.set_camera_location(self.camera_handles[-1], env, gymapi.Vec3(*camera_position), gymapi.Vec3(*camera_target))
            # print(camera_props.width, camera_props.height, camera_position, camera_target)
            # print(camera_props)

        input("Press Enter to continue...")
        if cfg.simulator.headless == False:
            # Look at the first env
            cam_pos = gymapi.Vec3(0.7, 2, 0)
            cam_target = gymapi.Vec3(0.5, 0.5, 0)
            self.gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
            # while not self.gym.query_viewer_has_closed(viewer):
            #     # check if we should update
            #     # step the physics
            #     self.gym.simulate(self.sim)
            #     self.gym.fetch_results(self.sim, True)


            #     # step rendering
            #     self.gym.step_graphics(self.sim)
            #     self.gym.draw_viewer(viewer, self.sim, False)
            #     self.gym.sync_frame_time(self.sim)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        return env


    def set_state(self, state):
        self.gym.set_actor_dof_states(self.env, self.object_handles[-1], state, gymapi.STATE_POS)
        return

    def get_image_observation(self, cam_idx = 0):

        rgb_filename = "test_image.png"
        H = self.cfg.simulator.camera_height[cam_idx]
        W = self.cfg.simulator.camera_width[cam_idx]
        # self.gym.write_camera_image_to_file(self.sim, self.env, self.camera_handles[-1], gymapi.IMAGE_COLOR, rgb_filename)
        cam_rgb = torch.tensor(self.gym.get_camera_image(self.sim, self.env, self.camera_handles[-1], gymapi.IMAGE_COLOR).reshape(H, W, 4))
        cam_depth = torch.tensor(self.gym.get_camera_image(self.sim, self.env, self.camera_handles[-1], gymapi.IMAGE_DEPTH))
        # Convert to a pillow image and write it to disk
        # normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
        # normalized_depth_image.save("depth_image.jpg")
        # rgb_image = im.fromarray(rgb_image, mode="RGB")
        # rgb_image.save("./rgb_image.jpg")
        return cam_rgb, cam_depth

    def step(self, action, substeps=15):
        # action is a 3D vector
        action = torch.tensor(action).view(1, 3)
        target = gymtorch.unwrap_tensor(action)
        self.gym.set_dof_position_target_tensor(self.sim, target)
        for i in range(substeps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            if self.cfg.simulator.headless == False:
                self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)
        self.steps += 1
        se2,SE3 = self.get_state()
        rgb_image, depth_image = self.get_image_observation()
        # np.savetxt(f"log/depth_image{self.steps}.txt", depth_image, fmt="%d")
        cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handles[-1]))))
        cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.env, self.camera_handles[-1]))
        
        pts = myutils.depth_image_to_point_cloud(self.task_name, rgb_image, depth_image, 1.0, cam_vinv, cam_proj, index=self.steps)
        # Convert to a pillow image and write it to disk
        if self.save_images:
            rgb_image = im.fromarray(np.array(rgb_image[:,:,:3]), mode="RGB")
            rgb_image.save(f"{self.save_path}/rgb_image{self.steps}.png")
            normalized_depth = -255.0*(np.array(depth_image)/np.min(np.array(depth_image) + 1e-4))
            normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
            normalized_depth_image.save(f"{self.save_path}/depth_image{self.steps}.png")
        if self.save_pts:
            with open(f'{self.save_path}/points{self.steps}.obj', 'w') as f:
                for i in range(pts.shape[0]):
                    f.write('v {} {} {} {} {} {}\n'.format(pts[i, 0], pts[i, 1], pts[i, 2], pts[i, 3], pts[i, 4], pts[i, 5]))


        # rgb_image = im.fromarray(np.array(rgb_image[:,:,:3]), mode="RGB")
        # rgb_image.save(f"log/rgb_image{self.steps}.png")

        # print("SE2: ", se2)
        # print("SE3: ", SE3)
        return pts, SE3
    def get_first(self):
        se2,SE3 = self.get_state()
        rgb_image, depth_image = self.get_image_observation()
        # np.savetxt(f"log/depth_image{self.steps}.txt", depth_image, fmt="%d")
        cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handles[-1]))))
        cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.env, self.camera_handles[-1]))
        task_name = self.task_name
        pts = myutils.depth_image_to_point_cloud(task_name, rgb_image, depth_image, 1.0, cam_vinv, cam_proj, index=self.steps)
        num_point = pts.shape[0]
        SE3 = SE3.expand(num_point, -1, -1)
        info = {
            'pts': pts, # [100, 6]
            'pose': SE3 # [100, 4, 4]
        }
        return info
    def get_observation(self):
        se2,SE3 = self.get_state()
        rgb_image, depth_image = self.get_image_observation()
        # np.savetxt(f"log/depth_image{self.steps}.txt", depth_image, fmt="%d")
        cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handles[-1]))))
        cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.env, self.camera_handles[-1]))
        
        pts = myutils.depth_image_to_point_cloud(self.task_name, rgb_image, depth_image, 1.0, cam_vinv, cam_proj, True, index=self.steps)
        # Convert to a pillow image and write it to disk
        if self.save_images:
            rgb_image = im.fromarray(np.array(rgb_image[:,:,:3]), mode="RGB")
            rgb_image.save(f"{self.save_path}/rgb_image{self.steps}.png")
            normalized_depth = -255.0*(np.array(depth_image)/np.min(np.array(depth_image) + 1e-4))
            normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
            normalized_depth_image.save(f"{self.save_path}/depth_image{self.steps}.png")
        if self.save_pts:
            with open(f'{self.save_path}/points{self.steps}.obj', 'w') as f:
                for i in range(pts.shape[0]):
                    f.write('v {} {} {} {} {} {}\n'.format(pts[i, 0], pts[i, 1], pts[i, 2], pts[i, 3], pts[i, 4], pts[i, 5]))
        return pts, SE3
    
def se2_to_SE3(se2):
    # [batch, x, z, theta] -> [batch, 4, 4]
    # given a 3D vector in se2(translation and rotation on x-z plane), convert it to a 4x4 matrix of SE3
    assert se2.shape[-1] == 3
    batch_size = se2.shape[0]
    se3 = torch.zeros(batch_size, 4, 4)
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

if __name__ == "__main__":
    cfg_path = 'config/task/toy2D.py'
    sys.path.append(os.path.dirname(cfg_path))
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    cfg = importlib.import_module(os.path.basename(cfg_path)[:-3]).get_cfg_defaults()
    num_trajs = 50
    steps_per_traj = 10
    target_motion = torch.tensor([0.0, 0.0, np.pi/2])
    env = ToyEnv2D(cfg, save_images=True, save_pts=True)
    print("Env initialized")  
    for traj in range(num_trajs):
        #! chenrui: 这里的逻辑是，先将三角形随机初始化位置，然后让它运动成一条等变的轨迹
        init_x = (np.random.rand() - 0.5) * 0.2
        init_z = (np.random.rand() - 0.5) * 0.2
        init_rot = (np.random.rand() - 0.5) * np.pi
        init_state = torch.tensor([init_x, init_z, init_rot])
        # init_state = torch.tensor([-0.0592492, 0.06098646, 1.2646412 ])
        print("init_state:", init_state)
        target_state = init_state + target_motion
        data = {}
        data['pts'] = []
        data['pose'] = []
        data['gt_action'] = []
        data['episode_ends'] = []
        data['init_state'] = []
        data['init_state'].append(np.array(init_state))
        env.set_state(init_state)
        
        action_list = list()
        for i in range(steps_per_traj):
            action = target_state * (i + 1) / (steps_per_traj)
            # observation data
            obs_pts, obs_pose = env.get_observation()
            print("obs_pose:", obs_pose)
            data['pts'].append(np.array(obs_pts))
            data['pose'].append(np.array(obs_pose).squeeze())
            print("!!!step_idx:",i)

            act_pts, act_pose = env.step(action)  #pts have the shape of (N, 6)
            # print("act_pose",act_pose)
            # exit(0)
            print("action",action)
            
            # print("act_pose.shape",act_pose.shape)
            print("SE2_act_pose",SE3_to_se2(act_pose))
            # print("after_pose",np.array(act_pose).squeeze())
            # exit(0)
            data['gt_action'].append(np.array(act_pose).squeeze())
            data['episode_ends'].append(np.array(steps_per_traj))
            # print(pts.shape)
            # print(pose.shape)
        data['pts'] = np.array(data['pts'])
        data['pose'] = np.array(data['pose'])
        data['gt_action'] = np.array(data['gt_action'])
        data['init_state'] = np.array(data['init_state'])
        # print("\n\n\ngt_action.shape",data['gt_action'].shape)
        # print("")
        # print("\n\n\npose_shape",data['pose'].shape)
        # print("pts_shape",data['pts'].shape)
        # print("pose",data['pose'])
        # print("pts",data['pts'])
        # print('action_list',action_list)
        # data['episode_ends'] = np.array(data['episode_ends'])
    np.save(f"log/{num_trajs}_rotate_triangle.npy", data)
        # exit(0)
        # bp()