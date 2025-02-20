"""Isaac default setting."""

from yacs.config import CfgNode as CN
from isaacgym import gymutil
_C = CN()
_C.name = 'isaac_default'

# Set True to run headless without creating a viewer window
_C.headless = True

# Disable graphics context creation, no viewer window is created, and no headless rendering is available
_C.nographics = False

_C.pipeline = "gpu"

# _C.physics_engine = "flex"
_C.physics_engine = "physx"
_C.num_threads = 0
_C.use_gpu = True
_C.sim_device = 'cuda:0'
_C.graphics_device_id = 0

_C.cameras = 1
_C.camera_width = [360]
_C.camera_height = [240]
_C.camera_position = [(0.6, 1.0, 0.05)]
_C.camera_target = [(0.65, 0.3, 0.05)]
_C.camera_fov = [175]


def get_cfg_defaults():
    sim_device_type, compute_device_id = gymutil.parse_device_str(_C.sim_device)
    _C.sim_device_type = sim_device_type
    _C.compute_device_id = compute_device_id
    return _C.clone()
