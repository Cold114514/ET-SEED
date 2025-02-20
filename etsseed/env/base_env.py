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

class BaseEnv():
    def __init__(self, cfg, task_name, num_points, save_path, save_images, save_pts):
        # set some basic parameters
        self.task_name = task_name
        self.cfg = cfg
        self.env = self.create_env(cfg)
        self.save_path = save_path + "/" + task_name + "/" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.save_path)
        self.save_images = save_images
        self.save_pts = save_pts
        self.num_points = num_points

    def create_env(self, cfg):
        #! This function creates the gym environment
        #! return: gym environment
        raise NotImplementedError

    def get_state(self):
        #! This function returns the SE(3) state of the desired object 
        #! return: A array of size (n, 4, 4) where n is the number of objects, each 4x4 matrix represents the SE(3) state of the object
        raise NotImplementedError

    def get_dof_state(self):
        #! This function returns the dof state of all DoFs in the environment
        #! return: tensor of size (num_dofs, 2). Each DOF state contains position and velocity.
        raise NotImplementedError

    def get_rigid_body_states(self):
        #! This function returns the rigid body states of all objects in the environment
        #! return: tensor of size (num_rigid_bodies, 13).
        #! State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        raise NotImplementedError

    def get_image(self, cam_idx):
        #! This function returns the rgb and depth images from the camera with index cam_idx
        #! return: cam_rgb - numpy array of size (H, W, 3), cam_depth - numpy array of size (H, W)
        raise NotImplementedError

    def get_pts(self):
        #! This function returns the point cloud of the environment
        #! return: numpy array of size (self.num_points, 3)
        raise NotImplementedError

    def step(self, action, substeps=1, if_log=False):
        #! This function takes an action and steps the environment
        #! action: A array of size (n, 4, 4) where n is the number of objects, each 4x4 matrix represents the SE(3) state of the object
        #! return: pointcloud - numpy array of size (self.num_points, 3)
        #!         SE3 state - numpy array of size (n, 4, 4)
        raise NotImplementedError

    def reset(self, frame_transformation):
        #! This function resets the environment with the 
        # ! desired object transformed by given transformation
        raise NotImplementedError

    def process_action(self, action):
        #! This function processes the raw action(different in tasks) before applying it to the environment
        #! return: processed action - numpy array of size (n, 4, 4) where n is the number of objects, each 4x4 matrix represents the SE(3) state of the object
        raise NotImplementedError