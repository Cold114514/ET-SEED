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
_C.task_name = 'door_opening'
_C.num_points = 1024
_C.frame_translation = [0., 0, 0.0]
_C.frame_rotation = [0, 0, 0] # represent in euler angles(rad)
def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    cfg.simulator.camera_width = [360]
    cfg.simulator.camera_height = [240]
    cfg.simulator.camera_position = [(0.3, 0.6, -0.8)]
    cfg.simulator.camera_target = [(0.65, 0.59, -0.05)]
    cfg.simulator.camera_fov = [90]
    cfg.simulator.headless = True
    return cfg
