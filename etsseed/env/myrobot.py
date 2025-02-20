# """
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# Franka Attractor
# ----------------
# Positional control of franka panda robot with a target attractor that the robot tries to reach
# """

# import math
# from isaacgym import gymapi
# from isaacgym import gymutil
# import numpy as np

# # Initialize gym
# gym = gymapi.acquire_gym()

# # Parse arguments
# args = gymutil.parse_arguments(description="Franka Attractor Example")

# # configure sim
# sim_type = args.physics_engine
# sim_params = gymapi.SimParams()
# sim_params.dt = 1.0 / 60.0
# sim_params.substeps = 2
# if args.physics_engine == gymapi.SIM_FLEX:
#     sim_params.flex.solver_type = 5
#     sim_params.flex.num_outer_iterations = 4
#     sim_params.flex.num_inner_iterations = 15
#     sim_params.flex.relaxation = 0.75
#     sim_params.flex.warm_start = 0.8
# elif args.physics_engine == gymapi.SIM_PHYSX:
#     sim_params.physx.solver_type = 1
#     sim_params.physx.num_position_iterations = 4
#     sim_params.physx.num_velocity_iterations = 1
#     sim_params.physx.num_threads = args.num_threads
#     sim_params.physx.use_gpu = args.use_gpu

# sim_params.use_gpu_pipeline = False
# if args.use_gpu_pipeline:
#     print("WARNING: Forcing CPU pipeline.")

# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# if sim is None:
#     print("*** Failed to create sim")
#     quit()

# plane_params = gymapi.PlaneParams()
# gym.add_ground(sim, plane_params)

# # Create viewer
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     print("*** Failed to create viewer")
#     quit()

# # Add ground plane
# plane_params = gymapi.PlaneParams()
# gym.add_ground(sim, plane_params)
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

import random
import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from copy import copy

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

tray_color = gymapi.Vec3(0.24, 0.35, 0.8)
banana_color = gymapi.Vec3(0.85, 0.88, 0.2)
brick_color = gymapi.Vec3(0.9, 0.5, 0.1)


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Kuka Bin Test",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
        {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
        {"name": "--object_type", "type": int, "default": 0, "help": "Type of bjects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"}])

num_envs = args.num_envs
num_objects = args.num_objects
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
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load assets
asset_root = "../../assets"

# Load franka asset
# asset_root = "../../assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
door_file = "8867/mobility.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

asset_options_robot = asset_options
franka_asset = gym.load_asset(
    sim, asset_root, franka_asset_file, asset_options_robot)

asset_options_door = asset_options
asset_options_door.flip_visual_attachments = False

print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
door_asset = gym.load_asset(
    sim, asset_root, door_file, asset_options_door)

# Set up the env grid
num_envs = 1
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
envs = []
franka_handles = []
doors = []
franka_hand = "panda_hand"

# Attractor setup
attractor_handles = []
attractor_properties = gymapi.AttractorProperties()
attractor_properties.stiffness = 5e5
attractor_properties.damping = 5e3

# Make attractor in all axes
attractor_properties.axes = gymapi.AXIS_ALL

# Create helper geometry used for visualization
# Create an wireframe axis
axes_geom = gymutil.AxesGeometry(0.1)
# Create an wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))


door_end = []
for i in range(num_envs):
    # create env
    
    franka_pose = gymapi.Transform()
    franka_pose.p = gymapi.Vec3(0, 0.0, 0.0)
    franka_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    door_pose = gymapi.Transform()
    door_pose.p = gymapi.Vec3(-0.3, 0.8, 0.5)
    door_pose.r = gymapi.Quat.from_euler_zyx(-math.pi/2, -math.pi/2, 0)
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    door = gym.create_actor(env, door_asset, door_pose, "door", i, 1)
    doors.append(door)
    # door_end.append(gym.find_actor_rigid_body_handle(env, door, "link_2"))

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 1)
    body_dict = gym.get_actor_rigid_body_dict(env, franka_handle)
    props = gym.get_actor_rigid_body_states(env, franka_handle, gymapi.STATE_POS)
    hand_handle = body = gym.find_actor_rigid_body_handle(env, franka_handle, franka_hand)

    # Initialize the attractor
    attractor_properties.target = props['pose'][:][body_dict[franka_hand]]
    attractor_properties.target.p.y -= 0.1
    attractor_properties.target.p.z = 0.1
    attractor_properties.rigid_handle = hand_handle

    # Draw axes and sphere at attractor location
    gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

    franka_handles.append(franka_handle)
    attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)
    attractor_handles.append(attractor_handle)


# get joint limits and ranges for Franka
franka_dof_props = gym.get_actor_dof_properties(envs[0], franka_handles[0])
franka_lower_limits = franka_dof_props['lower']
franka_upper_limits = franka_dof_props['upper']
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
franka_num_dofs = len(franka_dof_props)

# override default stiffness and damping values
franka_dof_props['stiffness'].fill(1000.0)
franka_dof_props['damping'].fill(1000.0)

# Give a desired pose for first 2 robot joints to improve stability
franka_dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS

franka_dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
franka_dof_props['stiffness'][7:] = 1e10
franka_dof_props['damping'][7:] = 1.0

for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], franka_handles[i], franka_dof_props)


def update_franka(position):
    gym.clear_lines(viewer)
    for i in range(num_envs):
        # Update attractor target from current franka state
        attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i])
        pose = attractor_properties.target
        print("curret pose : ", pose.p)
        pose.p.x = random.uniform(0, 0.5)
        # 0.2 * math.sin(1.5 * t - math.pi * float(i) / num_envs)
        pose.p.y = random.uniform(0, 0.5) # 0.7 + 0.1 * math.cos(2.5 * t - math.pi * float(i) / num_envs)
        pose.p.z = random.uniform(0, 0.5) # 0.2 * math.cos(1.5 * t - math.pi * float(i) / num_envs)
        pose.p = gymapi.Vec3(position[0], position[1], position[2])
        print("new pose : ", pose.p)

        gym.set_attractor_target(envs[i], attractor_handles[i], pose)

        # Draw axes and sphere at attractor location
        gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
        gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)


for i in range(num_envs):
    # Set updated stiffness and damping properties
    gym.set_actor_dof_properties(envs[i], franka_handles[i], franka_dof_props)

    # Set ranka pose so that each joint is in the middle of its actuation range
    franka_dof_states = gym.get_actor_dof_states(envs[i], franka_handles[i], gymapi.STATE_NONE)
    for j in range(franka_num_dofs):
        franka_dof_states['pos'][j] = franka_mids[j]
    gym.set_actor_dof_states(envs[i], franka_handles[i], franka_dof_states, gymapi.STATE_POS)

# Point camera at environments
cam_pos = gymapi.Vec3(-4.0, 2.0, -1.0)
cam_target = gymapi.Vec3(0.0, 1.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Time to wait in seconds before moving robot
next_franka_update_time = 0.3

door_state_props = gym.get_actor_dof_properties(envs[0], doors[0])
lower = door_state_props['lower']
upper = door_state_props['upper']
ranges = upper - lower
mids = 0.5 * (upper + lower)

for i in range(num_envs):
    # Set ranka pose so that each joint is in the middle of its actuation range
    door_state = gym.get_actor_dof_states(envs[i], doors[i], gymapi.STATE_NONE)
    door_state[0][0] = -6.668000e-01
    gym.set_actor_dof_states(envs[i], doors[i], door_state, gymapi.STATE_POS)

door_state_props['stiffness'].fill(1000.0)
door_state_props['damping'].fill(1000.0)
door_state_props["driveMode"][0:] = gymapi.DOF_MODE_POS

for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], doors[i], door_state_props)

# cur = lower[0]
# while cur <= upper[0]:
#     door_state = gym.get_actor_dof_states(envs[0], doors[0], gymapi.STATE_NONE)
#     print(door_state)
#     door_state[0][0] = cur
#     print("FFF  ",cur, door_state)
#     gym.set_actor_dof_states(envs[0], doors[0], door_state, gymapi.STATE_POS)
#     door_pose = gym.get_actor_rigid_body_states(envs[0], door, gymapi.STATE_POS)[3][0][0]
#     door_poses.append(door_pose)
#     cur += 0.05 * ranges[0]


# import copy
# poses = []
# while not gym.query_viewer_has_closed(viewer):
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, False)
#     gym.sync_frame_time(sim)
#     pos = gym.get_actor_rigid_body_states(envs[0], door, gymapi.STATE_POS)[3][0][0]
#     poses.append(copy.deepcopy(pos))
# poses = sorted(poses, key=lambda x: x[0], reverse=True)
# print(poses)

poses = [(0.0365006, 0.60561806, 0.6149704), (0.03532226, 0.6056181, 0.6431283), (0.03453378, 0.6056179, 0.6536756), (0.03453378, 0.6056179, 0.6536756), (0.03453378, 0.6056179, 0.6536756), (0.03453378, 0.6056179, 0.6536756), (0.03453378, 0.6056179, 0.6536756), (0.03281897, 0.6056179, 0.53764844), (0.02379547, 0.60561776, 0.48202664), (0.00514659, 0.6056182, 0.41414994), (-0.06330808, 0.60561794, 0.27559245), (-0.10133997, 0.6056178, 0.22490197), (-0.11049829, 0.6056177, 0.21423028), (-0.11525226, 0.60561764, 0.20888816), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525243, 0.60561764, 0.208888), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525244, 0.6056179, 0.20888773), (-0.11525256, 0.60561764, 0.20888785), (-0.11525256, 0.60561764, 0.20888785), (-0.11525256, 0.60561764, 0.20888785), (-0.11525256, 0.60561764, 0.20888785), (-0.16003987, 0.60561794, 0.16422392), (-0.22683722, 0.6056179, 0.11249784), (-0.28765112, 0.60561794, 0.0770481), (-0.39249003, 0.6056179, 0.03605076), (-0.91863626, 0.6056179, 0.13916767), (-0.92140436, 0.6056179, 0.14134908), (-0.92140436, 0.6056179, 0.14134908), (-0.92140436, 0.6056179, 0.14134908), (-0.92140436, 0.6056179, 0.14134908), (-0.92140436, 0.6056179, 0.14134908), (-0.92140436, 0.6056179, 0.14134908), (-0.92140436, 0.6056179, 0.14134908), (-0.92142075, 0.605618, 0.1413621), (-0.92142075, 0.605618, 0.1413621), (-0.92142075, 0.605618, 0.1413621), (-0.92142075, 0.605618, 0.1413621), (-0.92142075, 0.605618, 0.1413621), (-0.92142075, 0.605618, 0.1413621), (-0.92142075, 0.605618, 0.1413621)]
index = 0

while not gym.query_viewer_has_closed(viewer):
    # Every 0.01 seconds the pose of the attactor is updated
    t = gym.get_sim_time(sim)
    # print ("TIME : _------------------------------------ " , t, next_franka_update_time)
    if t >= next_franka_update_time:
        update_franka(poses[index])
        index = index + 2
        next_franka_update_time += 0.3
    

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    door_state = gym.get_actor_dof_states(envs[0], doors[0], gymapi.STATE_NONE)
    print(door_state)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
