"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Kuka bin perfromance test
-------------------------------
Test simulation perfromance and stability of the robotic arm dealing with a set of complex objects in a bin.
"""

from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from copy import copy
from pdb import set_trace as bp
import random

axes_geom = gymutil.AxesGeometry(0.1)

sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

colors = [gymapi.Vec3(1.0, 0.0, 0.0),
          gymapi.Vec3(1.0, 127.0/255.0, 0.0),
          gymapi.Vec3(1.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 0.0, 1.0),
          gymapi.Vec3(39.0/255.0, 0.0, 51.0/255.0),
          gymapi.Vec3(139.0/255.0, 0.0, 1.0)]


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="2D toy env",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
        ])

num_envs = args.num_envs
box_size = 0.05

# configure sim
sim_type = args.physics_engine
sim_params = gymapi.SimParams()
if sim_type == gymapi.SIM_FLEX:
    sim_params.substeps = 4
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 20
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif sim_type == gymapi.SIM_PHYSX:
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 25
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.rest_offset = 0.001

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
l_color = gymapi.Vec3(0.2,0.2,0.2)
l_ambient = gymapi.Vec3(0.5,0.5,0.5)
l_direction = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
gym.set_light_parameters(sim, 0, l_color, l_ambient, l_direction)
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load assets
asset_root = "../assets"

# table_dims = gymapi.Vec3(0.6, 0.4, 1.0)

# pose = gymapi.Transform()
# pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
# pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.001
asset_options.fix_base_link = True
asset_options.thickness = 0.002

asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

# table_pose = gymapi.Transform()
# table_pose.p = gymapi.Vec3(0.7, 0.5 * table_dims.y + 0.001, 0.0)

# bin_pose = gymapi.Transform()
# bin_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# load assets of objects in a bin
asset_options.fix_base_link = True

door_file = "41083/mobility.urdf"


object_assets = []


object_assets.append(gym.load_asset(sim, asset_root, door_file, asset_options))

# spawn_height = gymapi.Vec3(0.0, 0.03, 0.0)

# corner = table_pose.p - table_dims * 0.5

asset_root = "../assets"


if sim_type == gymapi.SIM_FLEX:
    asset_options.max_angular_velocity = 40.

# set up the env grid
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# cache some common handles for later use
envs = []
kuka_handles = []
tray_handles = []
object_handles = []
attractor_handles = {}

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
base_poses = []

# create env
env = gym.create_env(sim, env_lower, env_upper, num_per_row)
envs.append(env)

object_pose = gymapi.Transform()
object_pose.p = gymapi.Vec3(0, 1, 0)
object_pose.r = gymapi.Quat.from_euler_zyx(-math.pi/2, math.pi, 0)
# print(object_pose.p)
object_asset = object_assets[0]
triangle = gym.create_actor(env, object_asset, object_pose, "object" , 0, 0)
object_handles.append(triangle)
#! set object properties
shape_props = gym.get_actor_rigid_shape_properties(env, triangle)
shape_props[0].friction = 0.
shape_props[0].rolling_friction = 0.
shape_props[0].torsion_friction = 0.
gym.set_actor_rigid_shape_properties(env, triangle, shape_props)
gym.set_rigid_body_color(env, object_handles[-1], 3, gymapi.MESH_VISUAL_AND_COLLISION, colors[0])


#! set controller
# props = gym.get_actor_dof_properties(env, triangle)
# props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
# props["stiffness"] = (5000.0, 5000.0, 5000.0)
# props["damping"] = (100.0, 100.0, 100.0)
# print(props)
# gym.set_actor_dof_properties(env, triangle, props)

#! In this 2D setting, 3 DoF in total
# x_dof_handle = gym.find_actor_dof_handle(env, triangle, 'trans_x')
# z_dof_handle = gym.find_actor_dof_handle(env, triangle, 'trans_z')
# y_dof_handle = gym.find_actor_dof_handle(env, triangle, 'rot_y')
# gym.set_dof_target_position(env, x_dof_handle, 0)
# gym.set_dof_target_position(env, z_dof_handle, 0)
# gym.set_dof_target_position(env, y_dof_handle, 0.5)

frame = 0
# Look at the first env
cam_pos = gymapi.Vec3(1, 1, 0)
cam_target = gymapi.Vec3(0, 1, 0)
gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):
    # check if we should update

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

#    for env in envs:
#        gym.draw_env_rigid_contacts(viewer, env, colors[0], 0.5, True)

    # step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)



    x_dof_handle = gym.find_actor_dof_handle(env, triangle, 'trans_x')
    y_dof_handle = gym.find_actor_dof_handle(env, triangle, 'trans_y')
    z_dof_handle = gym.find_actor_dof_handle(env, triangle, 'trans_z')
    # gym.set_dof_target_position(env, x_dof_handle, 5)
    # gym.set_dof_target_position(env, y_dof_handle, 20)
    # gym.set_dof_target_position(env, z_dof_handle, 0)

    props = gym.get_actor_rigid_body_states(env, triangle, gymapi.STATE_POS)
    # print(props)
    # bp()
    frame = frame + 1

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
