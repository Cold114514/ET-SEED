import os, sys
from yacs.config import CfgNode as CN
from pathlib import Path

from myutils import merge_cfg

_base_ = {
    'simulator': '../simulator/isaac_default.py',
}

_C = CN()

_C.num_envs = 1
_C.task_name = 'rotate_triangle'
_C.num_points = 1024
_C.action_space = 3

def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    # cfg.simulator.headless = False
    return cfg
