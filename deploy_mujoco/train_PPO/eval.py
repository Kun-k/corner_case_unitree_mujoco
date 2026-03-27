import csv
import os
import pickle

import numpy as np
import yaml
from stable_baselines3 import PPO

from deploy.deploy_mujoco_go2.classifier_gate import ClassifierGate
from deploy.deploy_mujoco_go2.terrain_trainer import TerrainGymEnv, TerrainTrainer


def _build_log_paths(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return {
        "log_dir": log_dir,
        "pkl": os.path.join(log_dir, "collision_failures.pkl"),
        "non_failure_pkl": os.path.join(log_dir, "non_failure_trajectories.pkl"),
        "csv": os.path.join(log_dir, "failure_summary.csv"),
    }


def _append_csv_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episodes_evaluated",
                "total_failures",
                "collision_failures",
                "fall_failures",
                "base_collision_failures",
                "thigh_collision_failures",
                "stuck_failures",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_policy(
    model,
    go2_cfg,
    terrain_cfg,
    episodes,
    max_episode_steps,
    log_dir,
    seed,
    render,
    classifier_gate=None,
    classifier_threshold=0.5,
    save_non_failure_trajectories=False,
):
    current_path = os.path.dirname(os.path.realpath(__file__))

    trainer = TerrainTrainer(go2_cfg, terrain_cfg)
    if not trainer.render and render:
        trainer.render = True
        trainer.start_viewer()

    env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
    rng = np.random.default_rng(seed)

    paths = _build_log_paths(log_dir)

    go2_cfg_file = os.path.join(current_path, "../", go2_cfg[0], "configs", go2_cfg[1])
    terrain_cfg_file = os.path.join(current_path, "../", terrain_cfg)
    if os.path.exists(go2_cfg_file):
        os.system(f"cp {go2_cfg_file} {paths['log_dir']}")
    if os.path.exists(terrain_cfg_file):
        os.system(f"cp {terrain_cfg_file} {paths['log_dir']}")

    summary = {
        "episodes_evaluated": 0,
        "total_failures": 0,
        "collision_failures": 0,
        "fall_failures": 0,
        "base_collision_failures": 0,
        "thigh_collision_failures": 0,
        "stuck_failures": 0,
    }

    failures = []
    non_failures = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_chain = []

        has_collision = False
        has_fall = False
        has_base_collision = False
        has_thigh_collision = False
        has_stuck = False

        for step_idx in range(max_episode_steps):
            print(f"Episode {ep + 1}/{episodes}, Step {step_idx + 1}/{max_episode_steps}", end="\r")

            if model is None:
                action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            else:
                rl_action, _ = model.predict(obs, deterministic=False)
                rl_action = np.asarray(rl_action, dtype=np.float32)
                if classifier_gate is None:
                    action = rl_action
                else:
                    score = float(classifier_gate.predict_proba(np.asarray(obs, dtype=np.float32), rl_action)[0])
                    if score > float(classifier_threshold):
                        action = rl_action
                    else:
                        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            ep_chain.append(
                {
                    "obs": np.asarray(obs, dtype=np.float32).tolist(),
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                    "reward": float(reward),
                    "next_obs": np.asarray(next_obs, dtype=np.float32).tolist(),
                    "done": done,
                    "terrain_control": np.asarray(action, dtype=np.float32).tolist(),
                    "info": info,
                }
            )

            has_fall = has_fall or bool(info.get("fallen", False))
            has_base_collision = has_base_collision or bool(info.get("base_collision", False))
            has_thigh_collision = has_thigh_collision or bool(info.get("thigh_collision", False))
            has_stuck = has_stuck or bool(info.get("stuck", False))
            has_collision = has_collision or has_base_collision or has_thigh_collision

            obs = next_obs
            if done:
                break

        has_failure = has_collision or has_fall or has_stuck or has_base_collision or has_thigh_collision

        summary["episodes_evaluated"] += 1
        summary["total_failures"] += int(has_failure)
        summary["collision_failures"] += int(has_collision)
        summary["fall_failures"] += int(has_fall)
        summary["base_collision_failures"] += int(has_base_collision)
        summary["thigh_collision_failures"] += int(has_thigh_collision)
        summary["stuck_failures"] += int(has_stuck)

        if has_failure:
            failures.append({"episode": ep, "chain": ep_chain})
            with open(paths["pkl"], "wb") as pf:
                pickle.dump(failures, pf)
        elif save_non_failure_trajectories:
            non_failures.append({"episode": ep, "chain": ep_chain})
            with open(paths["non_failure_pkl"], "wb") as pf:
                pickle.dump(non_failures, pf)

        if summary["episodes_evaluated"] % 100 == 0:
            _append_csv_row(paths["csv"], summary.copy())

    _append_csv_row(paths["csv"], summary.copy())
    trainer.close_viewer()


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, "eval_config.yaml"), "r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)

    log_dir = os.path.join(current_path, "eval_logs", eval_config["log_name"])

    policy = eval_config["policy"]
    checkpoint = eval_config.get("checkpoint", "")

    if policy == "random":
        print("Evaluating random policy...")
        model = None
    else:
        model_path = str(os.path.join(current_path, "train_logs", policy, checkpoint))
        print(f"Evaluating PPO policy from {model_path}...")
        model = PPO.load(model_path)

    if policy == "random" or bool(eval_config.get("use_curr_force", False)):
        go2_cfg = [eval_config["go2_task"], eval_config["go2_config"]]
        terrain_cfg = f"train_PPO/{eval_config['terrain_config']}"
        episodes = int(eval_config["episodes"])
        max_episode_steps = int(eval_config["max_episode_steps"])
        seed = int(eval_config["seed"])
    else:
        train_log_dir = os.path.join(current_path, "train_logs", policy)
        train_config_file = os.path.join(train_log_dir, "train_config.yaml")
        with open(train_config_file, "r", encoding="utf-8") as f:
            train_config = yaml.safe_load(f)
        go2_cfg = [train_config["go2_task"], train_config["go2_config"]]
        terrain_cfg = f"train_PPO/train_logs/{policy}/terrain_config.yaml"
        episodes = int(eval_config["episodes"])
        max_episode_steps = int(train_config["max_episode_steps"])
        seed = 0

    classifier_gate = None
    gate_cfg = eval_config.get("classifier_gate", {})
    if bool(gate_cfg.get("enabled", False)):
        ckpt = str(gate_cfg.get("checkpoint_path", "")).strip()
        if not ckpt:
            raise ValueError("classifier_gate.enabled=true but checkpoint_path is empty")
        if not os.path.isabs(ckpt):
            ckpt = os.path.normpath(os.path.join(current_path, ckpt))
        classifier_gate = ClassifierGate(
            checkpoint_path=ckpt,
            device=str(gate_cfg.get("device", "cpu")),
            hidden_dim=int(gate_cfg.get("hidden_dim", 1024)),
            concat_action_to_obs=bool(gate_cfg.get("concat_action_to_obs", True)),
        )

    evaluate_policy(
        model=model,
        go2_cfg=go2_cfg,
        terrain_cfg=terrain_cfg,
        episodes=episodes,
        max_episode_steps=max_episode_steps,
        log_dir=log_dir,
        seed=seed,
        render=bool(eval_config.get("render", False)),
        classifier_gate=classifier_gate,
        classifier_threshold=float(gate_cfg.get("threshold", 0.5)),
        save_non_failure_trajectories=bool(eval_config.get("save_non_failure_trajectories", False)),
    )


if __name__ == "__main__":
    main()

