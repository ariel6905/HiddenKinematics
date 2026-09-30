"""IsaacGym environment: a Franka on a slider base opening one door or drawer per environment.

Derived from the PartManip environment (Geng et al., CVPR 2023).
"""
import json
from pathlib import Path

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.gymtorch import wrap_tensor
from pytorch3d.transforms import matrix_to_quaternion, quaternion_apply, quaternion_invert
from scipy.spatial.transform import Rotation

from .utils.misc import (CAMERAS, CAMERA_RESOLUTION, FRANKA_ASSET_OPTIONS, FRANKA_INIT_POS_DOOR,
                        FRANKA_INIT_POS_DRAWER, FRANKA_INIT_ROT, OBJECT_INIT_POS, OBJECT_INIT_POS_NP,
                        OBJECT_INIT_ROT, OBJECT_INIT_ROT_NP, PLANE_FRICTION, SEG_BASE, SEG_GRIPPER,
                        SEG_HANDLE, SEG_PART)
from .utils.perform_action import apply_actions
from .utils.compute import part_and_handle_bbox
from .utils.get_observation import observe
from .utils.get_reward import compute_reward


class CabinetEnv:
    """Vectorized door/drawer-opening environment."""

    def __init__(self, cfg, sim_params, assets, device_id=0):
        """``assets``: {"train": [...], "val": [...]} lists of asset directories
        (``.../<door|drawer>/<split>/<name>``); the category of each asset is read from its path."""
        self.cfg, self.sim_params = cfg, sim_params
        self.device_id, self.device = device_id, f"cuda:{device_id}"
        self.use_pc = cfg["point_cloud"] is not None
        self.max_episode_length = cfg["episode_length"]
        self.env_per_asset = cfg["env_per_asset"]

        self.selected_asset_path_list = [Path(p) for s in ("train", "val") for p in assets[s]]
        self.cabinet_num = len(self.selected_asset_path_list)
        self.env_num = self.num_envs = self.cabinet_num * self.env_per_asset
        self.env_num_train = len(assets["train"]) * self.env_per_asset        # Envs are ordered train, val
        self.is_door_cabinet = torch.tensor([p.parent.parent.name == "door" for p in self.selected_asset_path_list],
                                            device=self.device, dtype=torch.bool)
        self.is_door = self.is_door_cabinet.repeat_interleave(self.env_per_asset)
        op = cfg["open_proportion"]
        dc = cfg["reward"]["decouple"]
        self.decouple_coef = torch.where(self.is_door, dc["door"], dc["drawer"])
        self.num_actions = 8                                  # xyz + axis-angle + 2 fingers

        # Simulation
        self.gym = gymapi.acquire_gym()
        self.graphics_device_id = device_id if self.use_pc else -1     # -1 = headless; cameras need graphics
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.env_ptr_list, self.cabinet_actor_list = [], []
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        self._create_sim()
        self.gym.prepare_sim(self.sim)
        self._acquire_tensors()
        self.state_dim = self.franka_num_dofs * 2 + 2 + 16 + 24
        self.obs_dim = self.franka_num_dofs * 2 + 16 + 3
        if self.use_pc:
            for i, env_ptr in enumerate(self.env_ptr_list):
                for body in (self.hand_rigid_body_index, self.hand_lfinger_rigid_body_index, self.hand_rfinger_rigid_body_index):
                    self.gym.set_rigid_body_segmentation_id(env_ptr, self.franka_actor, body, SEG_GRIPPER)
                cab = self.cabinet_actor_list[i]
                self.gym.set_rigid_body_segmentation_id(env_ptr, cab, self.env_base_rigid_id_list[i], SEG_BASE)
                self.gym.set_rigid_body_segmentation_id(env_ptr, cab, self.env_part_rigid_id_list[i], SEG_PART)
                self.gym.set_rigid_body_segmentation_id(env_ptr, cab, self.env_handle_rigid_id_list[i], SEG_HANDLE)

        # Task targets: success when the joint reaches open_proportion (per category)
        self.cabinet_dof_target = torch.ones_like(self.dof_state_tensor_used[:, self.cabinet_dof_index, 0]) \
            * torch.where(self.is_door, op["door"], op["drawer"])
        self.success_dof_states = torch.where(self.is_door_cabinet, op["door"], op["drawer"]).to(self.device)
        self._refresh_state()
        self.success_buf = torch.zeros((self.env_num,), device=self.device).long()
        self.sparse_paid = torch.zeros((self.env_num,), device=self.device)
        self._init_canonical_frame()

    # ------------------------------------------------------------------ Construction
    def _create_sim(self):
        self.dt = self.sim_params.dt
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, gymapi.SIM_PHYSX, self.sim_params)
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0., 0., 1.)
        plane.static_friction = plane.dynamic_friction = PLANE_FRICTION
        self.gym.add_ground(self.sim, plane)
        if self.use_pc:
            self._init_camera_buffers()
        self._load_object_assets()
        spacing = self.cfg["env_spacing"]
        lower, upper = gymapi.Vec3(-spacing, -spacing, -spacing), gymapi.Vec3(spacing, spacing, spacing)
        per_row = int(np.sqrt(self.num_envs))
        self.franka_asset = None
        for env_id in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, per_row)
            self._create_franka(env_ptr, env_id)
            self._create_object(env_ptr, env_id)
            self.env_ptr_list.append(env_ptr)
            if self.use_pc:
                self._create_cameras(env_ptr, env_id)

    def _load_object_assets(self):
        self.cabinet_asset_list, self.cabinet_dof_num, self.rigid_num_list = [], [], []
        self.part_rigid_ids, self.handle_rigid_ids, self.part_dof_ids, self.base_rigid_ids = [], [], [], []
        part_bbox, handle_bbox, axis_xyz, axis_dir = [], [], [], []
        for path in self.selected_asset_path_list:
            link, handle, joint = path.name.split("-")[-4], path.name.split("-")[-3], path.name.split("-")[-2]
            opts = gymapi.AssetOptions()
            opts.fix_base_link = True
            opts.disable_gravity = True
            opts.collapse_fixed_joints = False
            asset = self.gym.load_asset(self.sim, str(path.absolute()), "mobility_new.urdf", opts)
            with open(path / "bbox_info.json") as f:
                bbox_info = json.load(f)
            dof_dict = self.gym.get_asset_dof_dict(asset)
            rig_names = self.gym.get_asset_rigid_body_dict(asset)
            self.cabinet_asset_list.append(asset)
            self.part_rigid_ids.append(rig_names[link])
            self.handle_rigid_ids.append(rig_names[handle])
            self.part_dof_ids.append(dof_dict[joint])
            self.base_rigid_ids.append(rig_names["base"])
            k = bbox_info["link_name"].index(link)
            part_bbox.append(np.array(bbox_info["bbox_world"][k]).astype(np.float32))
            handle_bbox.append(np.array(bbox_info["bbox_world"][bbox_info["link_name"].index(handle)]).astype(np.float32))
            axis_xyz.append(np.array(bbox_info["axis_xyz_world"][k]).astype(np.float32))
            axis_dir.append(np.array(bbox_info["axis_dir_world"][k]).astype(np.float32))
            self.cabinet_dof_num.append(self.gym.get_asset_dof_count(asset))
            self.rigid_num_list.append(len(rig_names))

        rep = lambda x: torch.tensor(np.array(x).astype(np.float32), device=self.device).repeat_interleave(self.env_per_asset, dim=0)
        self.part_bbox_tensor_init = rep(part_bbox)                # Rest-pose boxes / joint axis (object frame)
        self.handle_bbox_tensor_init = rep(handle_bbox)
        self.part_axis_xyz_tensor_init = rep(axis_xyz)
        self.part_axis_dir_tensor_init = rep(axis_dir)
        self.object_init_pose_p_tensor = torch.tensor(OBJECT_INIT_POS_NP.astype(np.float32), device=self.device)
        self.object_init_pose_r_matrix_tensor = torch.tensor(
            Rotation.from_quat(OBJECT_INIT_ROT_NP).as_matrix().astype(np.float32), device=self.device)
        ids = lambda x: torch.tensor(x, device=self.device).repeat_interleave(self.env_per_asset, dim=0)
        self.env_base_rigid_id_list = ids(self.base_rigid_ids)
        self.env_part_rigid_id_list = ids(self.part_rigid_ids)
        self.env_handle_rigid_id_list = ids(self.handle_rigid_ids)

    def _create_franka(self, env_ptr, env_id):
        if self.franka_asset is None:
            opts = gymapi.AssetOptions()
            for k, v in FRANKA_ASSET_OPTIONS.items():
                setattr(opts, k, v)
            self.franka_asset = self.gym.load_asset(self.sim, self.cfg["asset_root"], self.cfg["robot_urdf"], opts)
            props = self.gym.get_asset_dof_properties(self.franka_asset)
            self.franka_dof_lower_limits_tensor = torch.tensor(np.array(props["lower"]), device=self.device)
            self.franka_dof_upper_limits_tensor = torch.tensor(np.array(props["upper"]), device=self.device)
            self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
            self.franka_rigid_num = len(self.gym.get_asset_rigid_body_dict(self.franka_asset))
        dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][:].fill(400.0)
        dof_props["damping"][:].fill(40.0)
        # Finger drives get zero stiffness and damping, so finger position targets have no effect
        dof_props["stiffness"][-2:].fill(0)
        dof_props["damping"][-2:].fill(0)
        pose = gymapi.Transform()
        pose.r = FRANKA_INIT_ROT
        pose.p = FRANKA_INIT_POS_DOOR if bool(self.is_door[env_id]) else FRANKA_INIT_POS_DRAWER
        # Start from the pre-grasp configuration stored with the asset (gripper in front of the part)
        dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
        pregrasp = np.load(self.selected_asset_path_list[env_id // self.env_per_asset] / "part_pregrasp_dof_state.npy",
                           allow_pickle=True)
        dof_state["pos"] = pregrasp[:-1, 0]
        actor = self.gym.create_actor(env_ptr, self.franka_asset, pose, "franka", env_id, 2, 0)
        self.gym.set_actor_dof_properties(env_ptr, actor, dof_props)
        self.gym.set_actor_dof_states(env_ptr, actor, dof_state, gymapi.STATE_ALL)
        self.gym.set_actor_scale(env_ptr, actor, self.cfg["robot_scale"])
        self.franka_actor = actor

    def _create_object(self, env_ptr, env_id):
        cabinet = env_id // self.env_per_asset
        pose = gymapi.Transform()
        pose.p, pose.r = OBJECT_INIT_POS, OBJECT_INIT_ROT
        actor = self.gym.create_actor(env_ptr, self.cabinet_asset_list[cabinet], pose,
                                      f"cabinet{cabinet}-{env_id % self.env_per_asset}", env_id, 1, 0)
        props = self.gym.get_asset_dof_properties(self.cabinet_asset_list[cabinet])
        # Drive and friction settings go to DOF 0, which is not always the target joint
        props["stiffness"][0] = 20.0
        props["damping"][0] = 200
        props["friction"][0] = 5
        props["effort"][0] = 0.1
        props["driveMode"][0] = gymapi.DOF_MODE_NONE
        self.gym.set_actor_dof_properties(env_ptr, actor, props)
        self.cabinet_actor_list.append(actor)

    def _init_camera_buffers(self):
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = self.camera_props.height = CAMERA_RESOLUTION
        self.camera_props.enable_tensors = True
        self.camera_depth_tensor_list, self.camera_rgb_tensor_list, self.camera_seg_tensor_list = [], [], []
        self.camera_view_matrix_inv_list, self.camera_proj_matrix_list = [], []
        u = torch.arange(0, self.camera_props.width, device=self.device)
        v = torch.arange(0, self.camera_props.height, device=self.device)
        self.camera_v2, self.camera_u2 = torch.meshgrid(v, u, indexing="ij")
        self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

    def _create_cameras(self, env_ptr, env_id):
        depth, rgb, seg, view_inv, proj = [], [], [], [], []
        handles = [self.gym.create_camera_sensor(env_ptr, self.camera_props) for _ in CAMERAS]
        for h, (pos, target) in zip(handles, CAMERAS):
            self.gym.set_camera_location(h, env_ptr, pos, target)
        for image_type, out in ((gymapi.IMAGE_DEPTH, depth), (gymapi.IMAGE_COLOR, rgb), (gymapi.IMAGE_SEGMENTATION, seg)):
            for h in handles:
                out.append(gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, h, image_type)))
        for h in handles:
            view_inv.append(torch.inverse(torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, h))).to(self.device))
        for h in handles:
            proj.append(torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, h), device=self.device))
        self.camera_depth_tensor_list.append(depth)
        self.camera_rgb_tensor_list.append(rgb)
        self.camera_seg_tensor_list.append(seg)
        self.camera_view_matrix_inv_list.append(view_inv)
        self.camera_proj_matrix_list.append(proj)
        origin = self.gym.get_env_origin(env_ptr)
        self.env_origin[env_id][0], self.env_origin[env_id][1], self.env_origin[env_id][2] = origin.x, origin.y, origin.z

    def _acquire_tensors(self):
        """Wrap the simulator state and select the robot DOFs, the target joint and the
        (base, part, handle) rigid bodies of every env."""
        self._refresh_state()
        root = wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.jacobian_tensor = wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka"))
        self.dof_state_tensor_all = wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.rigid_body_tensor_all = wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.dof_state_tensor_used, self.dof_state_mask = _select_dofs(
            self.dof_state_tensor_all, self.part_dof_ids, self.cabinet_dof_num, self.franka_num_dofs,
            self.env_num, self.env_per_asset)
        self.rigid_body_tensor_used, self.rigid_state_mask = _select_bodies(
            self.rigid_body_tensor_all, self.part_rigid_ids, self.handle_rigid_ids, self.base_rigid_ids,
            self.rigid_num_list, self.franka_rigid_num, self.env_num, self.env_per_asset)
        self.root_tensor = root.view(self.num_envs, -1, 13)
        self.initial_dof_states = self.dof_state_tensor_all.clone()
        self.initial_root_states = self.root_tensor.clone()

        env_ptr = self.env_ptr_list[0]
        find = lambda name: self.gym.find_actor_rigid_body_index(env_ptr, self.franka_actor, name, gymapi.DOMAIN_ENV)
        self.hand_rigid_body_index = find("panda_hand")
        self.hand_lfinger_rigid_body_index = find("panda_leftfinger")
        self.hand_rfinger_rigid_body_index = find("panda_rightfinger")
        self.cabinet_dof_index = self.franka_num_dofs
        self.hand_rigid_body_tensor = self.rigid_body_tensor_used[:, self.hand_rigid_body_index, :]
        self.franka_dof_tensor = self.dof_state_tensor_used[:, :self.franka_num_dofs, :]
        self.cabinet_dof_tensor = self.dof_state_tensor_used[:, self.cabinet_dof_index, :]
        self.cabinet_dof_tensor_spec = self.cabinet_dof_tensor.view(self.cabinet_num, self.env_per_asset, -1)
        self.part_bbox_tensor, self.handle_bbox_tensor = part_and_handle_bbox(self, self.cabinet_dof_tensor[:, 0])
        self.init_part_bbox_tensor = self.part_bbox_tensor
        self.franka_root_tensor = self.root_tensor[:, 0, :]
        self.cabinet_root_tensor = self.root_tensor[:, 1, :]
        dof_dim = self.franka_num_dofs + 1
        self.pos_act = torch.zeros((self.num_envs, dof_dim), device=self.device)
        self.vel_act = torch.zeros((self.num_envs, dof_dim), device=self.device)
        self.eff_act = torch.zeros((self.num_envs, dof_dim), device=self.device)
        self.pos_act_all = torch.zeros_like(self.dof_state_tensor_all[:, 0], device=self.device)
        self.vel_act_all = torch.zeros_like(self.dof_state_tensor_all[:, 0], device=self.device)
        self.eff_act_all = torch.zeros_like(self.dof_state_tensor_all[:, 0], device=self.device)

    def _refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _init_canonical_frame(self):
        """Handle frame at reset, in which the observations are expressed."""
        hb = self.handle_bbox_tensor
        self.canon_center = (hb[:, 0, :] + hb[:, 6, :]) / 2
        out, long, short = hb[:, 0] - hb[:, 4], hb[:, 1] - hb[:, 0], hb[:, 3] - hb[:, 0]
        rot = torch.cat([(out / torch.norm(out, dim=1, keepdim=True)).view(-1, 1, 3),
                         (short / torch.norm(short, dim=1, keepdim=True)).view(-1, 1, 3),
                         (long / torch.norm(long, dim=1, keepdim=True)).view(-1, 1, 3)], dim=1)
        self.canon_quaternion_rot = matrix_to_quaternion(rot)
        self.canon_quaternion_rot_invert = quaternion_invert(self.canon_quaternion_rot)

    # ------------------------------------------------------------------ Interaction
    def step(self, actions):
        """Apply ``actions`` [E, 8] and simulate one step; returns ``(obs, reward, done, info)``."""
        apply_actions(self, actions)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if self.use_pc:
            self.gym.step_graphics(self.sim)
        observe(self)
        compute_reward(self)
        done = self.reset_buf.clone()
        info = {"successes": self.success_buf.clone()}
        self._reset_envs(self.reset_buf)
        self.progress_buf += torch.tensor([1], device=self.device)
        return self.obs_buf, self.rew_buf, done, info

    def reset(self):
        """Reset every env and return the first observation."""
        self._reset_envs(np.ones((self.env_num,)))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if self.use_pc:
            self.gym.step_graphics(self.sim)
        observe(self)
        compute_reward(self)
        return self.obs_buf

    def _reset_envs(self, to_reset):
        """Reset the flagged envs. Episodes have a fixed length, so every env ends at the same step
        and the whole simulator is restored to its initial state."""
        reset_any = False
        for env_id, flag in enumerate(to_reset):
            if flag.item():
                reset_any = True
                self.progress_buf[env_id] = 0
                self.reset_buf[env_id] = 0
                self.success_buf[env_id] = 0
                self.sparse_paid[env_id] = 0.0
        if reset_any:
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_root_states))

    def uncanonicalize(self, actions):
        """Map actions predicted in the handle frame (state teacher) back to the simulation frame."""
        a = actions.clone()
        a[:, :3] = quaternion_apply(self.canon_quaternion_rot_invert, a[:, :3])
        a[:, 3:6] = quaternion_apply(self.canon_quaternion_rot_invert, a[:, 3:6])
        return a


def _select_dofs(state, target_dof_ids, cabinet_dof_nums, franka_dofs, env_num, env_per_asset):
    """Select the robot DOFs and the target joint of each env from the flat simulator DOF tensor."""
    mask = torch.zeros((env_num, franka_dofs + 1), device=state.device)
    now = 0
    for i in range(env_num):
        a = i // env_per_asset
        mask[i, 0:franka_dofs] = torch.arange(now, now + franka_dofs, device=state.device)
        mask[i, franka_dofs] = now + franka_dofs + target_dof_ids[a]
        now += franka_dofs + cabinet_dof_nums[a]
    mask = mask.long()
    return state[mask], mask


def _select_bodies(state, part_ids, handle_ids, base_ids, cabinet_body_nums, franka_bodies, env_num, env_per_asset):
    """Select the robot bodies and (base, part, handle) of each env from the flat rigid-body tensor."""
    mask = torch.zeros((env_num, franka_bodies + 3), device=state.device)
    now = 0
    for i in range(env_num):
        a = i // env_per_asset
        mask[i, 0:franka_bodies] = torch.arange(now, now + franka_bodies, device=state.device)
        mask[i, franka_bodies] = now + franka_bodies + base_ids[a]
        mask[i, franka_bodies + 1] = now + franka_bodies + part_ids[a]
        mask[i, franka_bodies + 2] = now + franka_bodies + handle_ids[a]
        now += franka_bodies + cabinet_body_nums[a]
    mask = mask.long()
    return state[mask], mask
