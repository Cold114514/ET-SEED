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
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from base_env import BaseEnv


colors = [gymapi.Vec3(1.0, 0.0, 0.0),
          gymapi.Vec3(1.0, 127.0/255.0, 0.0),
          gymapi.Vec3(1.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 0.0, 1.0),
          gymapi.Vec3(39.0/255.0, 0.0, 51.0/255.0),
          gymapi.Vec3(139.0/255.0, 0.0, 1.0)]


class DoorEnv(BaseEnv):
    def __init__(self, cfg, task_name="door_opening", num_points=1024, save_path="log", save_images=True, save_pts=True):

        self.task_name = task_name
        self.cfg = cfg
        self.env = self.create_env(cfg)
        self.steps = 0
        self.save_path = save_path + "/" + task_name + "/" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.save_path)
        self.save_images = save_images
        self.save_pts = save_pts
    #! information of these APIs can be found in base_env.py
    def get_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        target_pose = self.gym.get_attractor_properties(self.env, self.attractor_handles[-1]).target
        SE3 = myutils.transform_to_SE3(target_pose)
        # print("state", SE3)
        return SE3

    def get_dof_state(self):
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        return dof_states

    def get_rigid_body_states(self):
        _rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rigid_body_states = gymtorch.wrap_tensor(_rigid_body_states)
        return rigid_body_states

    def get_image(self, cam_idx = 0):
        H = self.cfg.simulator.camera_height[cam_idx]
        W = self.cfg.simulator.camera_width[cam_idx]
        cam_rgb = torch.tensor(self.gym.get_camera_image(self.sim, self.env, self.camera_handles[-1], gymapi.IMAGE_COLOR).reshape(H, W, 4))
        cam_depth = torch.tensor(self.gym.get_camera_image(self.sim, self.env, self.camera_handles[-1], gymapi.IMAGE_DEPTH))
        return cam_rgb, cam_depth

    def get_pts(self):
        rgb_image, depth_image = self.get_image()
        cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handles[-1]))))
        cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.env, self.camera_handles[-1]))
        pts = myutils.depth_image_to_point_cloud(self.task_name, rgb_image, depth_image, 1.0, cam_vinv, cam_proj)
        return pts

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
        frame_translation = gymapi.Vec3(cfg.frame_translation[0], cfg.frame_translation[1], cfg.frame_translation[2])
        frame_rotation = gymapi.Quat.from_euler_zyx(cfg.frame_rotation[0], cfg.frame_rotation[1], cfg.frame_rotation[2])
        asset_root = "assets"
        self.frame_transformation = gymapi.Transform(p=frame_translation, r=frame_rotation)

        franka_asset_options = gymapi.AssetOptions()
        franka_asset_options.armature = 0.001
        franka_asset_options.fix_base_link = True
        franka_asset_options.flip_visual_attachments = True
        franka_asset_options.thickness = 0.002
        franka_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        # franka_asset_file = "urdf/franka_description/robots/franka_panda_collision.urdf"
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, franka_asset_options)
        
        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)
        franka_pose.r = gymapi.Quat(-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2)
        
        object_dims = gymapi.Vec3(0.5, 0.4, 0.5)
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.armature = 0.001
        object_asset_options.fix_base_link = True
        object_asset_options.flip_visual_attachments = False
        object_asset_options.thickness = 0.002
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_file = "8867/mobility.urdf"
        object_asset = self.gym.load_asset(self.sim, asset_root, object_file, object_asset_options)
        
        object_pose = object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(0.60, 0.70, 0) + self.frame_transformation.p
        object_pose.r = self.frame_transformation.r * gymapi.Quat.from_euler_zyx(-math.pi/2, 0, 0)


        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.franka_handles = []
        # create env
        env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        franka = self.gym.create_actor(env, franka_asset, franka_pose, "franka" , 0, 0)
        object_handle = self.gym.create_actor(env, object_asset, object_pose, "object", 0, 1)
        self.gym.set_actor_scale(env, object_handle, 0.75)
        self.gym.set_rigid_body_color(env,object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1., 1., 1.0))
        franka_dof_count = self.gym.get_actor_dof_count(env, franka)
        object_dof_count = self.gym.get_actor_dof_count(env, object_handle)
        franka_rigid_body_count = self.gym.get_actor_rigid_body_count(env, franka)
        object_rigid_body_count = self.gym.get_actor_rigid_body_count(env, object_handle)
        print("Franka dof count: ", franka_dof_count)
        print("Object dof count: ", object_dof_count)
        print("Franka rigid body count: ", franka_rigid_body_count)
        print("Object rigid body count: ", object_rigid_body_count)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        dof_state_tensor = self.get_dof_state()
        # dof_state_tensor[0:7] = 0.0
        dof_state_tensor[-2, 0] = -0.80
        dof_state_tensor = gymtorch.unwrap_tensor(dof_state_tensor)
        self.gym.set_dof_state_tensor(self.sim, dof_state_tensor)
        #! chenrui: 这里一定要get一下pose，不然pose不会刷新
        _ = self.gym.get_actor_rigid_body_states(env, franka, gymapi.STATE_POS)
        __ = self.gym.get_actor_rigid_body_states(env, object_handle, gymapi.STATE_POS)
        # print("dof state\n", self.get_dof_state())
        # print("rigid state\n: ", self.get_rigid_body_states())
        # bp()

        franka_hand = "panda_hand"
        self.franka_handles.append(franka)

        #! set controller
        props = self.gym.get_actor_dof_properties(env, franka)
        props['stiffness'].fill(1000.0)
        props['damping'].fill(1000.0)
        props["driveMode"][0:2] = gymapi.DOF_MODE_POS

        props["driveMode"][7:] = gymapi.DOF_MODE_POS
        props['stiffness'][7:] = 1e10
        props['damping'][7:] = 1.0
        self.gym.set_actor_dof_properties(env, franka, props)
        gymapi.RigidShapeProperties.friction = 100000
        # props = self.gym.get_actor_dof_properties(env, object_handle)
        # props['stiffness'].fill(1000.0)
        # props['damping'].fill(1000.0)
        # props["driveMode"][:] = gymapi.DOF_MODE_POS
        # self.gym.set_actor_dof_properties(env, object_handle, props)

        # Attractor setup
        self.attractor_handles = []
        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 5e5
        attractor_properties.damping = 5e3
        offset = gymapi.Transform()
        # offset.p = self.frame_transformation.r.rotate(gymapi.Vec3(0.0, 0.0, 0.0))
        offset.p = gymapi.Vec3(-0.01, 0.01, 0.1)
        # offset.p = gymapi.Vec3(0.0, 0.01, 0.11)
        offset.r = gymapi.Quat.from_euler_zyx(0, 90, 0)
        attractor_properties.offset = offset

        # Make attractor in all axes
        attractor_properties.axes = gymapi.AXIS_ALL

        body_dict = self.gym.get_actor_rigid_body_dict(env, franka)
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=(1, 0, 0))
        #! chenrui: 这里一定要get一下pose，不然pose不会刷新
        props = self.gym.get_actor_rigid_body_states(env, franka, gymapi.STATE_POS)

        hand_handle = body = self.gym.find_actor_rigid_body_handle(env, franka, franka_hand)

        attractor_target = gymapi.Transform()
        handle_index = self.gym.find_actor_rigid_body_index(env, object_handle, 'link2', gymapi.DOMAIN_SIM)
        handle_pose = self.get_rigid_body_states()[handle_index]
        attractor_properties.target = gymapi.Transform()
        attractor_properties.target.p = gymapi.Vec3(*handle_pose[0:3])
        attractor_properties.target.r = self.frame_transformation.r * gymapi.Quat(*handle_pose[3:7])
        attractor_properties.rigid_handle = hand_handle
        self.init_attr_pos = copy(attractor_properties.target)
        attractor_handle = self.gym.create_rigid_body_attractor(env, attractor_properties)
        self.attractor_handles.append(attractor_handle)


        self.camera_handles = []
        #! set camera
        for cam in range(cfg.simulator.cameras):
            camera_props = gymapi.CameraProperties()
            camera_props.width = cfg.simulator.camera_width[cam]
            camera_props.height = cfg.simulator.camera_height[cam]
            camera_props.horizontal_fov = cfg.simulator.camera_fov[cam]
            camera_position = cfg.simulator.camera_position[cam]
            camera_target = cfg.simulator.camera_target[cam]
            self.camera_handles.append(self.gym.create_camera_sensor(env, camera_props))
            self.gym.set_camera_location(self.camera_handles[-1], env, gymapi.Vec3(*camera_position), gymapi.Vec3(*camera_target))
            print(camera_props.width, camera_props.height, camera_position, camera_target)
            # print(camera_props)

        # input("Press Enter to continue...")
        if cfg.simulator.headless == False:
            # Look at the first env
            cam_pos = gymapi.Vec3(0.2, 0.5, -1)
            cam_target = gymapi.Vec3(0.7, 0.5, -0.2)
            self.gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        return env


    def step(self, action, substeps=10, if_log=False):
        # self.gym.clear_lines(self.viewer)
        attractor_properties = self.gym.get_attractor_properties(self.env, self.attractor_handles[-1])
        #! action should be 4x4 matrix
        pose = attractor_properties.target
        if action is not None:
            pose = myutils.SE3_to_transform(action)
            if self.cfg.simulator.headless == False:
                gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.env, pose)
        self.gym.set_attractor_target(self.env, self.attractor_handles[-1], pose)
        # gymutil.draw_lines(self.axes_geom, self.gym, self.viewer,self.env, pose)
        print("Attractor pose: ", pose.p.x, pose.p.y, pose.p.z)
        franka = self.franka_handles[0]
        body_dict = self.gym.get_actor_rigid_body_dict(self.env, franka)
        for i in range(substeps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        if self.cfg.simulator.headless == False:
            self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.render_all_camera_sensors(self.sim)
        self.steps += 1
        SE3 = self.get_state()
        # np.savetxt(f"log/depth_image{self.steps}.txt", depth_image, fmt="%d")
        pts = self.get_pts()
        rgb_image, depth_image = self.get_image()
        if self.save_images and if_log:
            rgb_image = im.fromarray(np.array(rgb_image[:,:,:3]), mode="RGB")
            rgb_image.save(f"{self.save_path}/rgb_image{self.steps}.png")
            # -inf implies no depth value, set it to zero. output will be black.
            depth_image[depth_image == -np.inf] = 0

             # clamp depth image to 10 meters to make output image human friendly
            depth_image[depth_image < -10] = -10
            normalized_depth = -255.0*(np.array(depth_image)/np.min(np.array(depth_image) + 1e-4))
            normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
            normalized_depth_image.save(f"{self.save_path}/depth_image{self.steps}.png")
        if self.save_pts and if_log:
            with open(f'{self.save_path}/points{self.steps}.obj', 'w') as f:
                for i in range(pts.shape[0]):
                    f.write('v {} {} {} {} {} {}\n'.format(pts[i, 0], pts[i, 1], pts[i, 2], pts[i, 3], pts[i, 4], pts[i, 5]))
        return pts, SE3
    
    def process_action(self, action):
        #! Here action is a 6d vector(xyz,rpy), we need to convert it to SE3
        attractor_properties = self.gym.get_attractor_properties(self.env, self.attractor_handles[-1])
        action_rotation = gymapi.Quat.from_euler_zyx(action[3], action[4], action[5])
        pose = attractor_properties.target
        action_trans = self.frame_transformation.r.rotate(gymapi.Vec3(*action[:3]))
        pose.p.x = action_trans.x + self.init_attr_pos.p.x
        pose.p.y = action_trans.y + self.init_attr_pos.p.y
        pose.p.z = action_trans.z + self.init_attr_pos.p.z
        pose.r = action_rotation * self.init_attr_pos.r
        SE3 = myutils.transform_to_SE3(pose)
        return SE3


if __name__ == "__main__":
    cfg_path = 'config/task/door.py'
    sys.path.append(os.path.dirname(cfg_path))
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    cfg = importlib.import_module(os.path.basename(cfg_path)[:-3]).get_cfg_defaults()
    env = DoorEnv(cfg, save_images=True, save_pts=True)
    print("Env initialized")
    traj_length = 200
    actions = np.zeros((traj_length, 6))
    theta = 45 / 180 * np.pi
    for i in range(traj_length):
        interval = theta/traj_length
        actions[i, 0] = -np.sin(i*interval) * 0.36
        actions[i, 2] = (1 - np.cos(i*interval)) * 0.36
        actions[i, 4] = i * interval / 3

    save_interval = 10
    num_steps = np.shape(actions)[0]
    # warm up, drive the robot to the initial position
    #! 这几步是为了让机械臂到达初始位置，不计入正式轨迹
    print("Warm up")
    for _ in range(30):
        pts, pose = env.step(None)  #pts have the shape of (N, 6)
        current_pose = env.get_state()
    print("Warm up done")
    data = {}
    data['pts'] = []
    data['pose'] = []
    data['gt_action'] = []
    data['episode_ends'] = []
    obs_pts = env.get_pts()
    data['pts'].append(np.array(obs_pts))
    data['pose'].append(env.get_state())
    for i in range(num_steps):
        if_log = (((i+1) % save_interval) == 0) or (i == num_steps-1)
        obs_pts = env.get_pts()
        action = actions[i]
        SE3action = env.process_action(action)
        pts, pose = env.step(SE3action, if_log=if_log)
        if (if_log):
            data['gt_action'].append(env.get_state())
            #! chenrui: 注意这里是有相位差的，由于最开始存下了pts和pose，第j个pose对应第j+1个action
            data['pts'].append(np.array(obs_pts))
            data['pose'].append(env.get_state())
            data['episode_ends'].append(np.array(10))
            print('--------------------------------')
            print(f"Step {i} done")
            print("pose\n", data['pose'][-2])
            print("gt_action\n", data['gt_action'][-1])
            # bp()
    #! chenrui: 相位差，最后多存了一个pose和pts，需要删除
    data['pts'].pop(-1)
    data['pose'].pop(-1)

    data['pts'] = np.array(data['pts'])
    data['pose'] = np.array(data['pose'])
    data['gt_action'] = np.array(data['gt_action'])
    data['episode_ends'] = np.array(data['episode_ends'])
    for key in data.keys():
        print(key, data[key].shape)
    np.save(f"{env.save_path}/data.npy", data)
