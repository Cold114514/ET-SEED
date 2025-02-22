## Simulator
This isaac_default.py file is used to define default configuration settings for an Isaac Gym - based simulation. It employs the yacs library to create a structured configuration object, allowing easy management and adjustment of simulation parameters.

* headless: When set to True, the simulation runs without creating a viewer window.
* nographics: When set to True, graphics context creation is disabled, and no headless rendering is available.
* pipeline: Specifies the pipeline type, here set to use the GPU.
* physics_engine: Defines the physics engine to be used, with options like "flex" or "physx", currently set to "physx".
* num_threads: Sets the number of threads for the simulation.
* use_gpu: Determines whether to use the GPU for simulation.
* sim_device: Specifies the device for simulation, here set to the first CUDA device.
* graphics_device_id: Sets the ID of the graphics device.
* cameras: The number of cameras in the simulation.
* camera_width and camera_height: Define the resolution of the cameras.
* camera_position and camera_target: Specify the position and target of the cameras.
* camera_fov: Sets the field - of - view of the cameras.


## Task
Here is the task configuration for the simulation environment:

* _base_ Defines a base configuration file for the simulator. 
* _C is the main configuration node created using CfgNode.
* num_envs: The number of environments to run simultaneously. Here, it is set to 1.
* task_name: The name of the task, which is 'painting' in this case.
* num_points: The number of points involved in the task, set to 1024.
* action_space: The dimensionality of the action space, set to 3.
* frame_translation: A list representing the translation of the frame in 3D space.
* frame_rotation: A list representing the rotation of the frame in Euler angles (in radians).
* trajectory: Specifies the type of trajectory for the task. Options include 'star', 'heart', or 'calligraphy', and here it is set to 'calligraphy'.