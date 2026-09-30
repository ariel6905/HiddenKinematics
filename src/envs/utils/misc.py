"""Fixed simulation, robot and camera constants of the PartManip cabinet environment."""
import numpy as np
from isaacgym import gymapi

CLIP_ACTIONS = 3.0

# Simulator
SIM_DT = 1. / 60.
PHYSX = dict(solver_type=1, num_position_iterations=8, num_velocity_iterations=8, num_threads=8,
             max_gpu_contact_pairs=8 * 1024 * 1024, rest_offset=0.0, bounce_threshold_velocity=0.2,
             max_depenetration_velocity=1000.0, default_buffer_size_multiplier=5.0, contact_offset=1e-3)
PLANE_FRICTION = 0.1

# Robot (Franka on a slider base); quaternions are xyzw
FRANKA_ASSET_OPTIONS = dict(flip_visual_attachments=True, fix_base_link=True, disable_gravity=True, armature=0.01)
FRANKA_INIT_ROT = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
FRANKA_INIT_POS_DOOR = gymapi.Vec3(0.8, 0.0, 0.4)
FRANKA_INIT_POS_DRAWER = gymapi.Vec3(0.8, 0.0, 0.5)
IK_DAMPING = 0.05

# Articulated object (quaternions xyzw, as in gymapi / scipy)
OBJECT_INIT_POS_NP = np.array([-1, 0, 1.4])
OBJECT_INIT_ROT_NP = np.array([0., 0., 1., 0.])
OBJECT_INIT_POS = gymapi.Vec3(-1, 0, 1.4)
OBJECT_INIT_ROT = gymapi.Quat(0., 0., 1., 0.)

# The three scene cameras as (position, target)
CAMERAS = ((gymapi.Vec3(0.5, 0, 2.0), gymapi.Vec3(-2.0, 0., -0.8)),       # Front, looking down
           (gymapi.Vec3(0.0, 0.8, 2.3), gymapi.Vec3(-3.0, -2.0, -0.2)),   # Left
           (gymapi.Vec3(0.0, -0.8, 2.3), gymapi.Vec3(-3.0, 2.0, -0.2)))   # Right
CAMERA_RESOLUTION = 256

# Point-cloud segmentation labels (last channel of obs.points: x, y, z, r, g, b, seg)
SEG_BASE, SEG_PART, SEG_HANDLE, SEG_GRIPPER = 0, 1, 2, 3
