import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from deploy_mujoco.utils import get_gravity_orientation, pd_control, quat_to_heading_w, wrap_to_pi


class G1Controller:
    def __init__(self, config_file: str = "g1.yaml"):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        cfg_path = os.path.join(base_dir, "configs", config_file)
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self._project_root = os.path.normpath(os.path.join(base_dir, "../.."))
        self.policy_path = self._resolve_path(str(self.config["policy_path"]))
        self.xml_path = self._resolve_path(str(self.config["xml_path"]))

        self.simulation_duration = float(self.config["simulation_duration"])
        self.simulation_dt = float(self.config["simulation_dt"])
        self.control_decimation = int(self.config["control_decimation"])
        self.lock_camera = bool(self.config.get("lock_camera", True))
        self.phase_period = float(self.config.get("phase_period", 0.8))

        self.kps = np.array(self.config["kps"], dtype=np.float32)
        self.kds = np.array(self.config["kds"], dtype=np.float32)
        self.default_angles = np.array(self.config["default_angles"], dtype=np.float32)

        self.ang_vel_scale = float(self.config["ang_vel_scale"])
        self.dof_pos_scale = float(self.config["dof_pos_scale"])
        self.dof_vel_scale = float(self.config["dof_vel_scale"])
        self.action_scale = float(self.config["action_scale"])
        self.cmd_scale = np.array(self.config["cmd_scale"], dtype=np.float32)
        self.heading_command = bool(self.config.get("heading_command", False))
        self.heading_target = float(self.config.get("heading_target", 0.0))
        self.heading_stiffness = float(self.config.get("heading_stiffness", 1.0))

        self.num_actions = int(self.config["num_actions"])
        self.num_obs = int(self.config["num_obs"])

        self.cmd = np.array(self.config["cmd_init"], dtype=np.float32)
        self.action_policy_prev = np.zeros(self.num_actions, dtype=np.float32)

        self._validate_config()

        self.policy = torch.jit.load(self.policy_path)
        print(f"Loaded G1 policy from {self.policy_path}")

    def _resolve_path(self, path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(self._project_root, path_value))

    def _validate_config(self) -> None:
        if self.kps.shape[0] != self.num_actions:
            raise ValueError(f"kps length {self.kps.shape[0]} != num_actions {self.num_actions}")
        if self.kds.shape[0] != self.num_actions:
            raise ValueError(f"kds length {self.kds.shape[0]} != num_actions {self.num_actions}")
        if self.default_angles.shape[0] != self.num_actions:
            raise ValueError(
                f"default_angles length {self.default_angles.shape[0]} != num_actions {self.num_actions}"
            )
        expected_obs = 9 + 3 * self.num_actions + 2
        if self.num_obs != expected_obs:
            raise ValueError(f"num_obs {self.num_obs} != expected {expected_obs} for current G1 observation layout")

    def reset(self) -> None:
        self.action_policy_prev = np.zeros(self.num_actions, dtype=np.float32)

    def get_observation(self, d: mujoco.MjData, counter: int) -> np.ndarray:
        qj = d.qpos[7:].copy()
        dqj = d.qvel[6:].copy()
        quat = d.qpos[3:7].copy()
        omega = d.qvel[3:6].copy()

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega = omega * self.ang_vel_scale

        count_s = counter * self.simulation_dt
        phase = (count_s % self.phase_period) / max(self.phase_period, 1e-6)
        sin_phase = np.sin(2.0 * np.pi * phase)
        cos_phase = np.cos(2.0 * np.pi * phase)

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = omega
        obs[3:6] = gravity_orientation
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9: 9 + self.num_actions] = qj
        obs[9 + self.num_actions: 9 + 2 * self.num_actions] = dqj
        obs[9 + 2 * self.num_actions: 9 + 3 * self.num_actions] = self.action_policy_prev
        obs[9 + 3 * self.num_actions: 9 + 3 * self.num_actions + 2] = np.array([sin_phase, cos_phase], dtype=np.float32)
        return obs

    def update_command(self, d: mujoco.MjData) -> None:
        if not self.heading_command:
            return
        current_heading = quat_to_heading_w(d.qpos[3:7])
        heading_err = wrap_to_pi(self.heading_target - current_heading)
        self.cmd[2] = np.clip(heading_err * self.heading_stiffness, -1.0, 1.0)

    def compute_action(self, d: mujoco.MjData, counter: int) -> np.ndarray:
        self.update_command(d)
        obs = self.get_observation(d, counter)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action_policy = self.policy(obs_tensor).detach().cpu().numpy().squeeze()

        self.action_policy_prev[:] = action_policy.astype(np.float32)
        return self.action_policy_prev * self.action_scale + self.default_angles

    def get_observation_without_prev_action(self, d: mujoco.MjData, counter: int) -> np.ndarray:
        qj = d.qpos[7:].copy()
        dqj = d.qvel[6:].copy()
        quat = d.qpos[3:7].copy()
        omega = d.qvel[3:6].copy()

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega = omega * self.ang_vel_scale

        count_s = counter * self.simulation_dt
        phase = (count_s % self.phase_period) / max(self.phase_period, 1e-6)
        sin_phase = np.sin(2.0 * np.pi * phase)
        cos_phase = np.cos(2.0 * np.pi * phase)

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = omega
        obs[3:6] = gravity_orientation
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9: 9 + self.num_actions] = qj
        obs[9 + self.num_actions: 9 + 2 * self.num_actions] = dqj
        obs[9 + 2 * self.num_actions: 9 + 2 * self.num_actions + 2] = np.array([sin_phase, cos_phase], dtype=np.float32)
        return obs

    def run(self) -> None:
        m = mujoco.MjModel.from_xml_path(self.xml_path)
        d = mujoco.MjData(m)
        m.opt.timestep = self.simulation_dt

        target_dof_pos = self.default_angles.copy()

        with mujoco.viewer.launch_passive(m, d) as viewer:
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.5
            viewer.cam.lookat[:] = d.qpos[:3]

            counter = 1
            while counter % self.control_decimation != 0:
                tau = pd_control(target_dof_pos, d.qpos[7:], self.kps, np.zeros_like(self.kds), d.qvel[6:], self.kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                counter += 1

            start = time.time()
            while viewer.is_running() and (time.time() - start) < self.simulation_duration:
                step_start = time.time()

                if self.lock_camera:
                    viewer.cam.lookat[:] = d.qpos[:3]

                if counter % self.control_decimation == 0:
                    target_dof_pos = self.compute_action(d, counter)

                tau = pd_control(target_dof_pos, d.qpos[7:], self.kps, np.zeros_like(self.kds), d.qvel[6:], self.kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

                counter += 1
                viewer.sync()

                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

