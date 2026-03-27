import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from deploy.deploy_mujoco_go2.utils import (
    get_gravity_orientation,
    pd_control,
    quat_to_heading_w,
    wrap_to_pi,
)


def update_command(data, cmd, heading_stiffness, heading_target):
    """Always use heading-based yaw control for navigation."""
    current_heading = quat_to_heading_w(data.qpos[3:7])
    heading_err = wrap_to_pi(heading_target - current_heading)
    cmd[2] = np.clip(heading_err * heading_stiffness, -1.0, 1.0)
    return cmd


def sample_random_goal(cfg, rng):
    xr = cfg["goal_random"]["x_range"]
    yr = cfg["goal_random"]["y_range"]
    x = float(rng.uniform(float(xr[0]), float(xr[1])))
    y = float(rng.uniform(float(yr[0]), float(yr[1])))
    return np.array([x, y], dtype=np.float32), None


def get_goal_from_list(cfg, idx):
    goal_entry = cfg["goal_list"][idx]
    gx = float(goal_entry[0])
    gy = float(goal_entry[1])
    gheading = float(goal_entry[2]) if len(goal_entry) >= 3 else None
    return np.array([gx, gy], dtype=np.float32), gheading


def select_next_goal(cfg, goal_idx, rng):
    mode = str(cfg["goal_mode"]).lower()
    if mode == "random":
        return sample_random_goal(cfg, rng), goal_idx

    if mode != "list":
        raise ValueError(f"Unsupported goal_mode: {cfg['goal_mode']}")

    n = len(cfg["goal_list"])
    if n == 0:
        raise ValueError("goal_mode=list but goal_list is empty")

    if goal_idx is None:
        next_idx = 0
    else:
        next_idx = goal_idx + 1
        if next_idx >= n:
            if bool(cfg.get("goal_list_loop", True)):
                next_idx = 0
            else:
                next_idx = n - 1

    return get_goal_from_list(cfg, next_idx), next_idx


def compute_navigation_cmd(data, goal_xy, goal_heading, cfg):
    """Compute cmd = [vx_body_norm, vy_body_norm, wz_norm] from current pose and goal."""
    pos_xy = np.array(data.qpos[:2], dtype=np.float32)
    delta = goal_xy - pos_xy
    dist = float(np.linalg.norm(delta))

    # World-frame velocity command from simple proportional position control.
    kp_pos = float(cfg["goal_kp_pos"])
    vx_w = kp_pos * float(delta[0])
    vy_w = kp_pos * float(delta[1])

    max_vx = float(cfg["goal_max_lin_vel_x"])
    max_vy = float(cfg["goal_max_lin_vel_y"])
    vx_w = float(np.clip(vx_w, -max_vx, max_vx))
    vy_w = float(np.clip(vy_w, -max_vy, max_vy))

    # Convert world velocity command to body frame.
    yaw = float(quat_to_heading_w(data.qpos[3:7]))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w

    cmd = np.zeros(3, dtype=np.float32)
    cmd[0] = np.clip(vx_b / max(max_vx, 1e-6), -1.0, 1.0)
    cmd[1] = np.clip(vy_b / max(max_vy, 1e-6), -1.0, 1.0)

    # Desired heading: explicit goal heading (if provided), otherwise face goal point.
    if goal_heading is None:
        goal_heading = float(np.arctan2(delta[1], delta[0])) if dist > 1e-6 else float(quat_to_heading_w(data.qpos[3:7]))
    else:
        goal_heading = float(goal_heading)

    kp_heading = float(cfg["goal_kp_heading"])
    heading_target = yaw + kp_heading * (float(goal_heading) - yaw)

    return cmd, heading_target, dist


def maybe_switch_goal(cfg, dist, reached_count, goal_idx, rng):
    """Switch goal when robot remains within threshold for a few control cycles."""
    reach_dist = float(cfg["goal_reach_dist"])
    hold_steps = int(cfg["goal_reach_hold_steps"])

    if dist <= reach_dist:
        reached_count += 1
    else:
        reached_count = 0

    if reached_count >= hold_steps:
        (goal_xy, goal_heading), goal_idx = select_next_goal(cfg, goal_idx, rng)
        reached_count = 0
        return goal_xy, goal_heading, goal_idx, reached_count, True

    return None, None, goal_idx, reached_count, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the configs folder")
    args = parser.parse_args()

    cfg_path = f"{os.path.dirname(os.path.realpath(__file__))}/configs/{args.config_file}"
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = cfg["policy_path"]
    xml_path = cfg["xml_path"]
    simulation_duration = float(cfg["simulation_duration"])
    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])

    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)

    lin_vel_scale = float(cfg["lin_vel_scale"])
    ang_vel_scale = float(cfg["ang_vel_scale"])
    dof_pos_scale = float(cfg["dof_pos_scale"])
    dof_vel_scale = float(cfg["dof_vel_scale"])
    action_scale = float(cfg["action_scale"])
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)

    num_actions = int(cfg["num_actions"])
    num_obs = int(cfg["num_obs"])

    # heading_command is fixed True for this navigation deployment.
    heading_stiffness = float(cfg["heading_stiffness"])

    rng = np.random.default_rng(int(cfg.get("goal_random_seed", 0)))

    # Runtime states.
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    # Load model and policy.
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    policy = torch.jit.load(policy_path)

    # Initialize goal.
    goal_idx = None
    (goal_xy, goal_heading), goal_idx = select_next_goal(cfg, goal_idx, rng)
    reached_count = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()

        # Warmup to align control boundary.
        counter = 1
        while counter % control_decimation != 0:
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter += 1

        print(f"[terrain_navigation] start goal: ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")

        while viewer.is_running() and time.time() - start < simulation_duration:
            if counter % control_decimation == 0:
                cmd, heading_target, dist = compute_navigation_cmd(d, goal_xy, goal_heading, cfg)
                cmd = update_command(d, cmd, heading_stiffness, heading_target)
                print(cmd)

                # Build policy observation.
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                ang_vel = d.qvel[3:6]
                lin_vel = d.qvel[0:3]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                lin_vel = lin_vel * lin_vel_scale
                ang_vel = ang_vel * ang_vel_scale

                obs[0:3] = lin_vel
                obs[3:6] = ang_vel
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = qj
                obs[24:36] = dqj
                obs[36:48] = action

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

                new_goal_xy, new_goal_heading, goal_idx, reached_count, switched = maybe_switch_goal(
                    cfg, dist, reached_count, goal_idx, rng
                )
                if switched:
                    goal_xy = new_goal_xy
                    goal_heading = new_goal_heading
                    print(f"[terrain_navigation] switch goal -> ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter += 1
            viewer.sync()


if __name__ == "__main__":
    main()

