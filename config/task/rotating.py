import os, sys
from yacs.config import CfgNode as CN
from pathlib import Path
import numpy as np
from myutils import merge_cfg

_base_ = {
    'simulator': '../simulator/isaac_default.py',
}

_C = CN()

_C.num_envs = 1
_C.task_name = 'cap_rotating'
_C.num_points = 512
_C.action_space = 3
_C.frame_translation = [0, 0, 0.2]
_C.frame_rotation = [-np.pi/6, 0, 0] # represent in euler angles(rad)
# _C.frame_rotation = [0, 0, 0]
_C.trajectory = 'rotating'
def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    cfg.simulator.headless = False
    cfg.simulator.cameras = 1
    cfg.simulator.camera_width = [360]
    cfg.simulator.camera_height = [240]
    cfg.simulator.camera_position = [(0.5, 1.0, 0.0)]
    cfg.simulator.camera_target = [(0.58, 0.3, 0.2)]
    cfg.simulator.camera_fov = [160]
    return cfg
