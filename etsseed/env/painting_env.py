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
import argparse

class PaintingEnv():
    def __init__(self, cfg, task_name="painting", num_points=1024, save_path="log", save_images=True, save_pts=True):

        self.task_name = task_name
        self.cfg = cfg
        self.env = self.create_env(cfg)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.steps = 0
        self.save_path = save_path + "/" + task_name + "/" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.save_path)
        self.save_images = save_images
        self.save_pts = save_pts

    def get_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        target_pose = self.gym.get_attractor_properties(self.env, self.attractor_handles[-1]).target
        SE3 = myutils.transform_to_SE3(target_pose)
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
        #! chenrui: 不知道这里为什么开了viewer就不能拍点云
        if self.cfg.simulator.headless == False:
            pts = None
        else:
            pts = myutils.depth_image_to_point_cloud_painting(rgb_image, depth_image, 1.0, cam_vinv, cam_proj)
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
        asset_root = "assets"
        table_dims = gymapi.Vec3(0.5, 0.4, 0.5)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.thickness = 0.002

        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        table_pose = gymapi.Transform()
        frame_translation = gymapi.Vec3(cfg.frame_translation[0], cfg.frame_translation[1], cfg.frame_translation[2])
        frame_rotation = gymapi.Quat.from_euler_zyx(cfg.frame_rotation[0], cfg.frame_rotation[1], cfg.frame_rotation[2])
        self.frame_transformation = gymapi.Transform(p=frame_translation, r=frame_rotation)
        table_pose = self.frame_transformation
        table_pose.p += gymapi.Vec3(0.6, 0.0, 0.0)
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        self.particle_asset = self.gym.create_sphere(self.sim, 0.01, asset_options)

        # franka_asset_file = "urdf/franka_description/robots/franka_panda_brush.urdf"
        franka_asset_file = "urdf/franka_description/robots/franka_brush_collision.urdf"
        franka_assets = []
        franka_assets.append(self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options))

        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.franka_handles = []
        # create env
        env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", 1, 0)
        self.gym.set_rigid_body_color(env,table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1., 1., 1.0))
        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)
        franka_pose.r = gymapi.Quat(-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2)
        franka_asset = franka_assets[0]
        franka = self.gym.create_actor(env, franka_asset, franka_pose, "franka" , 0, 0)
        self.gym.set_rigid_body_color(env,franka, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0., 0., 0.0))

        franka_hand = "brush"
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
        
        # Attractor setup
        self.attractor_handles = []
        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 5e5
        attractor_properties.damping = 5e3
        offset = gymapi.Transform()
        # offset.p = self.frame_transformation.r.rotate(gymapi.Vec3(0.0, 0.0, 0.0))
        offset.p = gymapi.Vec3(0.0, 0.0, 0.14)
        offset.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        attractor_properties.offset = offset

        # Make attractor in all axes
        attractor_properties.axes = gymapi.AXIS_ALL

        # Create helper geometry used for visualization
        # Create an wireframe axis
        self.axes_geom = gymutil.AxesGeometry(0.1)
        # Create an wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=(1, 0, 0))

        body_dict = self.gym.get_actor_rigid_body_dict(env, franka)
        props = self.gym.get_actor_rigid_body_states(env, franka, gymapi.STATE_POS)
        hand_handle = body = self.gym.find_actor_rigid_body_handle(env, franka, franka_hand)

        attractor_target = gymapi.Transform()
        table_surface_center = table_pose.p + self.frame_transformation.r.rotate(gymapi.Vec3(0, table_dims.y/2, 0))
        attractor_properties.target = gymapi.Transform()
        attractor_properties.target.p = table_surface_center
        attractor_properties.target.r = self.frame_transformation.r * props['pose'][:][body_dict[franka_hand]][1]
        attractor_properties.rigid_handle = hand_handle
        self.init_attr_pos = copy(attractor_properties.target)
        # Draw axes and sphere at attractor location
        # gymutil.draw_lines(self.axes_geom, self.gym, viewer, env, attractor_properties.target)
        # if cfg.simulator.headless == False:
        #     gymutil.draw_lines(self.sphere_geom, self.gym, viewer, env, attractor_properties.target)
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
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(0, 0., 0.)
            local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(180.0))
            self.gym.attach_camera_to_body(self.camera_handles[-1], env, hand_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            # self.gym.set_camera_location(self.camera_handles[-1], env, gymapi.Vec3(*camera_position), gymapi.Vec3(*camera_target))
            # print(camera_props.width, camera_props.height, camera_position, camera_target)
            # print(camera_props)

        # input("Press Enter to continue...")
        if cfg.simulator.headless == False:
            # Look at the first env
            cam_pos = gymapi.Vec3(0.7, 2, 0)
            cam_target = gymapi.Vec3(0.5, 0.5, 0)
            self.gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        self.particle_handles = []
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
            self.particle_handles.append(self.gym.create_actor(self.env, self.particle_asset, pose, "particle", 1, 0))
            self.gym.set_rigid_body_color(self.env,self.particle_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1., 0., 0.0))
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
        #! Here action is a 3d vector, we need to convert it to SE3
        attractor_properties = self.gym.get_attractor_properties(self.env, self.attractor_handles[-1])
        pose = attractor_properties.target
        action = self.frame_transformation.r.rotate(gymapi.Vec3(*action))
        pose.p.x = action.x + self.init_attr_pos.p.x
        pose.p.y = action.y + self.init_attr_pos.p.y
        pose.p.z = action.z + self.init_attr_pos.p.z
        SE3 = myutils.transform_to_SE3(pose)
        return SE3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate painting data with different trajectories')
    parser.add_argument('--trajectory', type=str, default='star', 
                        choices=['star', 'heart', 'calligraphy'], 
                        help='Type of trajectory to generate (star, heart, calligraphy)')
    parser.add_argument('--save_dir', type=str, default='log/demo', 
                        help='Base directory for saving data (default: log/demo)')
    args = parser.parse_args()

    # Set up the save path dynamically
    save_base_dir = args.save_dir
    save_dir = os.path.join(save_base_dir, args.trajectory)
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_path = os.path.join(save_dir, f"{args.trajectory}_data.npy")
        
    cfg_path = 'config/task/painting.py'
    sys.path.append(os.path.dirname(cfg_path))
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    cfg = importlib.import_module(os.path.basename(cfg_path)[:-3]).get_cfg_defaults()
    assert cfg.trajectory in ['star', 'heart', 'calligraphy']
    env = PaintingEnv(cfg, save_pts=True)
    print("Env initialized")
    if cfg.trajectory == 'star':
        key_points = myutils.draw_5PointedStar(r = 0.15)
        steps_per_traj = 50
        actions2d = myutils.geometry_interpolation(key_points, steps_per_traj, is_closed=True)

    elif cfg.trajectory == 'calligraphy':
        #! chenrui: 这里暂时可选的字有： 对客挥毫秦少游
        actions2d = myutils.calligraphy(pinyin='qin')
    # key_points = myutils.draw_HeartCurve(a = 0.25, num_keypoints = 32)

    save_interval = 5
    num_steps = np.shape(actions2d)[0]
    actions = np.concatenate([actions2d[:,0][:,None], np.zeros((num_steps, 1)), actions2d[:,1][:,None]], axis=1)

    # warm up, drive the robot to the initial position
    #! 这几步是为了让机械臂到达初始位置，不计入正式轨迹
    for _ in range(10):
        pts, pose = env.step(None)  #pts have the shape of (N, 6)
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
    #! chenrui: 相位差，最后多存了一个pose和pts，需要删除
    data['pts'].pop(-1)
    data['pose'].pop(-1)

    data['pts'] = np.array(data['pts'])
    data['pose'] = np.array(data['pose'])
    data['gt_action'] = np.array(data['gt_action'])
    data['episode_ends'] = np.array(data['episode_ends'])
    for key in data.keys():
        print(key, data[key].shape)
    os.makedirs(args.save_path, exist_ok=True)
    np.save(os.path.join(args.save_path, f"{traj_type}_data.npy"), data)
    np.save(f"{env.save_path}/data.npy", data)

