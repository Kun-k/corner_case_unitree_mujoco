import os
import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import gymnasium as gym
except ImportError:
    import gym

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer, TerrainGymEnv
from deploy.deploy_mujoco_go2.classifier_gate import ClassifierGate
from deploy.deploy_mujoco_go2.offline_data_utils import collect_pkl_files, load_chains_from_pkl_file, filter_chain_for_replay
from deploy.deploy_mujoco_go2.reward_recompute_utils import (
    load_reward_cfg_from_yaml,
    recompute_reward_from_info,
)


def _extract_reward_cfg_from_terrain_yaml(terrain_cfg_file: str):
    return load_reward_cfg_from_yaml(terrain_cfg_file)


def _recompute_reward_from_info(info: dict, reward_cfg: dict) -> float:
    return recompute_reward_from_info(info, reward_cfg)


class FailureRecordingWrapper(gym.Wrapper):
    """Record failure episodes as transition chains and periodically dump to PKL."""

    def __init__(self, env, out_dir: str, pkl_name: str = "train_failure_chains.pkl", flush_every_episodes: int = 50):
        super().__init__(env)
        self.out_dir = out_dir
        self.pkl_path = os.path.join(out_dir, pkl_name)
        self.flush_every_episodes = int(max(1, flush_every_episodes))

        self._curr_obs = None
        self._curr_chain = []
        self._episode_idx = 0
        self._failure_episodes = []

        os.makedirs(self.out_dir, exist_ok=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._curr_obs = np.asarray(obs, dtype=np.float32)
        self._curr_chain = []
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        # info_dict = {
        #     "fallen": bool(info.get("fallen", False)),
        #     "collided": bool(info.get("collided", False)),
        #     "base_collision": bool(info.get("base_collision", False)),
        #     "thigh_collision": bool(info.get("thigh_collision", False)),
        #     "stuck": bool(info.get("stuck", False)),
        #     "terrain_reward": float(info.get("terrain_reward", 0.0)),
        # }

        tr = {
            "obs": np.asarray(self._curr_obs, dtype=np.float32).tolist() if self._curr_obs is not None else [],
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32).tolist(),
            "done": done,
            "terrain_control": np.asarray(action, dtype=np.float32).tolist(),
            "info": info,
        }
        self._curr_chain.append(tr)
        self._curr_obs = np.asarray(next_obs, dtype=np.float32)

        if done:
            has_failure = bool(
                info["fallen"]
                # or info_dict["collided"]
                or info["base_collision"]
                or info["thigh_collision"]
                or info["stuck"]
            )
            if has_failure:
                self._failure_episodes.append({"episode": int(self._episode_idx), "chain": self._curr_chain})
            self._episode_idx += 1

            if self._episode_idx % self.flush_every_episodes == 0:
                self._flush()

        return next_obs, reward, terminated, truncated, info

    def _flush(self):
        with open(self.pkl_path, "wb") as f:
            pickle.dump(self._failure_episodes, f)

    def close(self):
        self._flush()
        return self.env.close()


class TrainingLoggerCallback(BaseCallback):
    def __init__(
        self,
        out_dir: str,
        save_every_steps: int = 2000,
        smooth_window: int = 20,
        checkpoint_every_steps: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.save_every_steps = int(max(1, save_every_steps))
        self.smooth_window = int(max(1, smooth_window))
        self.checkpoint_every_steps = int(max(1, checkpoint_every_steps))

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timesteps = []
        self.actor_losses = []
        self.critic_losses = []
        self.loss_timesteps = []

        os.makedirs(self.out_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(float(info["episode"].get("l", 0)))
                self.episode_timesteps.append(int(self.num_timesteps))

        if hasattr(self.model, "logger"):
            log_dict = self.model.logger.name_to_value
            wrote_loss = False
            if "train/actor_loss" in log_dict:
                self.actor_losses.append(float(log_dict["train/actor_loss"]))
                wrote_loss = True
            if "train/critic_loss" in log_dict:
                self.critic_losses.append(float(log_dict["train/critic_loss"]))
                wrote_loss = True
            if wrote_loss:
                self.loss_timesteps.append(int(self.num_timesteps))

        if self.num_timesteps % self.save_every_steps == 0:
            self._dump_metrics_csv("metrics_partial.csv")
            self._save_combined_plot("training_curves_partial.png")

        if self.num_timesteps % self.checkpoint_every_steps == 0:
            ckpt_path = os.path.join(self.out_dir, f"checkpoint_step_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)

        return True

    def _on_training_end(self) -> None:
        self._save_plot(self.episode_rewards, "Episode Reward", "episode_reward.png")
        if len(self.actor_losses) > 0:
            self._save_plot(self.actor_losses, "Actor Loss", "actor_loss.png")
        if len(self.critic_losses) > 0:
            self._save_plot(self.critic_losses, "Critic Loss", "critic_loss.png")

        self._dump_metrics_csv("metrics.csv")
        self._save_combined_plot("training_curves.png")

    def _smooth(self, values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size < self.smooth_window:
            return arr
        kernel = np.ones((self.smooth_window,), dtype=np.float32) / float(self.smooth_window)
        return np.convolve(arr, kernel, mode="valid")

    def _save_plot(self, values, title: str, filename: str):
        if len(values) == 0:
            return
        fig = plt.figure()
        plt.plot(values, linewidth=1.0)
        smoothed = self._smooth(values)
        if smoothed.size > 0 and smoothed.size != len(values):
            offset = len(values) - smoothed.size
            plt.plot(np.arange(offset, len(values)), smoothed, linewidth=2.0)
        plt.title(title)
        plt.grid(True)
        fig.savefig(os.path.join(self.out_dir, filename), dpi=140)
        plt.close(fig)

    def _dump_metrics_csv(self, filename: str):
        csv_path = os.path.join(self.out_dir, filename)
        max_len = max(len(self.episode_rewards), len(self.actor_losses), len(self.critic_losses))
        if max_len == 0:
            return

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_timestep",
                "episode_reward",
                "episode_length",
                "loss_timestep",
                "actor_loss",
                "critic_loss",
            ])
            for i in range(max_len):
                writer.writerow([
                    self.episode_timesteps[i] if i < len(self.episode_timesteps) else "",
                    self.episode_rewards[i] if i < len(self.episode_rewards) else "",
                    self.episode_lengths[i] if i < len(self.episode_lengths) else "",
                    self.loss_timesteps[i] if i < len(self.loss_timesteps) else "",
                    self.actor_losses[i] if i < len(self.actor_losses) else "",
                    self.critic_losses[i] if i < len(self.critic_losses) else "",
                ])

    def _save_combined_plot(self, filename: str):
        if len(self.episode_rewards) == 0 and len(self.actor_losses) == 0 and len(self.critic_losses) == 0:
            return

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        if len(self.episode_rewards) > 0:
            axes[0].plot(self.episode_timesteps, self.episode_rewards, linewidth=1.0, label="reward")
            smoothed = self._smooth(self.episode_rewards)
            if smoothed.size > 0 and smoothed.size != len(self.episode_rewards):
                offset = len(self.episode_rewards) - smoothed.size
                axes[0].plot(self.episode_timesteps[offset:], smoothed, linewidth=2.0, label=f"reward_ma{self.smooth_window}")
            axes[0].set_title("Episode Reward")
            axes[0].grid(True)
            axes[0].legend(loc="best")

        if len(self.actor_losses) > 0:
            x_actor = self.loss_timesteps[:len(self.actor_losses)] if len(self.loss_timesteps) >= len(self.actor_losses) else np.arange(len(self.actor_losses))
            axes[1].plot(x_actor, self.actor_losses, linewidth=1.0, label="actor_loss")
            axes[1].set_title("Actor Loss")
            axes[1].grid(True)
            axes[1].legend(loc="best")

        if len(self.critic_losses) > 0:
            x_critic = self.loss_timesteps[:len(self.critic_losses)] if len(self.loss_timesteps) >= len(self.critic_losses) else np.arange(len(self.critic_losses))
            axes[2].plot(x_critic, self.critic_losses, linewidth=1.0, label="critic_loss")
            axes[2].set_title("Critic Loss")
            axes[2].grid(True)
            axes[2].legend(loc="best")

        for ax in axes:
            ax.set_xlabel("Timesteps")

        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, filename), dpi=150)
        plt.close(fig)


class FilteredGatedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, consecutive_fail_keep_k: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.consecutive_fail_keep_k = int(max(0, consecutive_fail_keep_k))
        self._pending = [[] for _ in range(self.n_envs)]

    def _select_episode_transitions(self, episode_chain):
        def _extra_keep_fn(tr, _idx):
            info = tr.get("info", {}) if isinstance(tr, dict) else {}
            return bool(info.get("use_rl_action", True)) if isinstance(info, dict) else True

        return filter_chain_for_replay(
            episode_chain,
            consecutive_fail_keep_k=int(self.consecutive_fail_keep_k),
            extra_keep_fn=_extra_keep_fn,
        )

    def _flush_episode(self, env_idx: int):
        ep = self._pending[env_idx]
        if not ep:
            return

        selected = self._select_episode_transitions(ep)
        for tr in selected:
            super().add(
                np.expand_dims(tr["obs"], axis=0),
                np.expand_dims(tr["next_obs"], axis=0),
                np.expand_dims(tr["action"], axis=0),
                np.array([tr["reward"]], dtype=np.float32),
                np.array([float(bool(tr["done"]))], dtype=np.float32),
                [tr["info"]],
            )

        self._pending[env_idx] = []

    def add(self, obs, next_obs, action, reward, done, infos):
        if not isinstance(infos, (list, tuple)):
            infos = [{} for _ in range(self.n_envs)]

        obs = np.asarray(obs)
        next_obs = np.asarray(next_obs)
        action = np.asarray(action)
        reward = np.asarray(reward).reshape(-1)
        done = np.asarray(done).reshape(-1)

        for env_idx in range(self.n_envs):
            info = dict(infos[env_idx]) if env_idx < len(infos) and isinstance(infos[env_idx], dict) else {}
            tr = {
                "obs": obs[env_idx].copy(),
                "next_obs": next_obs[env_idx].copy(),
                "action": action[env_idx].copy(),
                "reward": float(reward[env_idx]),
                "done": float(done[env_idx]),
                "info": info,
            }
            self._pending[env_idx].append(tr)

            if bool(done[env_idx]):
                self._flush_episode(env_idx)


class GatedSAC(SAC):
    """SAC variant that gates action source using a classifier score threshold.

    - score > threshold: use SAC action
    - else: use random action
    Only SAC-selected transitions are written to replay buffer.
    """

    def __init__(self, *args, gate: ClassifierGate, gate_threshold: float = 0.5, min_rl_buffer_size_for_update: int = 1000, consecutive_fail_keep_k: int = 0, **kwargs):
        kwargs.setdefault("replay_buffer_class", FilteredGatedReplayBuffer)
        rb_kwargs = dict(kwargs.get("replay_buffer_kwargs", {}))
        rb_kwargs.setdefault("consecutive_fail_keep_k", int(consecutive_fail_keep_k))
        kwargs["replay_buffer_kwargs"] = rb_kwargs
        kwargs["learning_starts"] = 0
        super().__init__(*args, **kwargs)
        self.gate = gate
        self.gate_threshold = float(gate_threshold)
        self.min_rl_buffer_size_for_update = int(max(1, min_rl_buffer_size_for_update))
        self._last_use_rl_mask = None
        self.rl_action_count = 0
        self.random_action_count = 0

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(0, action_noise=action_noise, n_envs=n_envs)

        probs = self.gate.predict_proba(self._last_obs, action)
        use_rl_mask = probs > self.gate_threshold
        self._last_use_rl_mask = np.asarray(use_rl_mask, dtype=np.bool_).reshape(-1)

        if np.any(~self._last_use_rl_mask):
            random_actions = np.stack([self.action_space.sample() for _ in range(int(np.sum(~self._last_use_rl_mask)))], axis=0).astype(np.float32)
            action[~self._last_use_rl_mask] = random_actions
            if isinstance(self.action_space, gym.spaces.Box):
                buffer_action[~self._last_use_rl_mask] = self.policy.scale_action(random_actions)
            else:
                buffer_action[~self._last_use_rl_mask] = random_actions

        self.rl_action_count += int(np.sum(self._last_use_rl_mask))
        self.random_action_count += int(np.sum(~self._last_use_rl_mask))
        return action, buffer_action

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        if isinstance(infos, (list, tuple)):
            mask = self._last_use_rl_mask
            if mask is None or len(mask) != len(infos):
                mask = np.ones((len(infos),), dtype=np.bool_)
            infos = [dict(info) if isinstance(info, dict) else {} for info in infos]
            for i in range(len(infos)):
                infos[i]["use_rl_action"] = bool(mask[i])
                infos[i]["gate_score"] = float(-1.0)
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def train(self, gradient_steps: int, batch_size: int = 64):
        upper_bound = self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos
        if upper_bound < self.min_rl_buffer_size_for_update:
            return
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)


def configure_torch_runtime(cfg: dict):
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    deterministic = bool(cfg.get("torch_deterministic", False))
    cudnn_benchmark = bool(cfg.get("cudnn_benchmark", True))
    allow_tf32 = bool(cfg.get("allow_tf32", True))

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    num_threads = int(cfg.get("torch_num_threads", 0))
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def preload_replay_buffer_from_pkl(model, pkl_paths, reward_cfg=None, consecutive_fail_keep_k: int = 0):
    files = collect_pkl_files(pkl_paths)
    inserted = 0

    obs_shape = model.observation_space.shape
    act_shape = model.action_space.shape

    for fp in files:
        try:
            chains = load_chains_from_pkl_file(fp, consecutive_fail_keep_k=0)
        except Exception:
            continue

        for chain in chains:
            filtered_chain = filter_chain_for_replay(chain, consecutive_fail_keep_k=int(consecutive_fail_keep_k))
            for tr in filtered_chain:
                try:
                    obs = np.asarray(tr.get("obs", []), dtype=np.float32).reshape(obs_shape)
                    next_obs = np.asarray(tr.get("next_obs", []), dtype=np.float32).reshape(obs_shape)
                    action = np.asarray(tr.get("action", []), dtype=np.float32).reshape(act_shape)
                    info = tr.get("info", {}) if isinstance(tr.get("info", {}), dict) else {}
                    reward = _recompute_reward_from_info(info, reward_cfg or {})
                    if not (reward_cfg or {}):
                        reward = float(tr.get("reward", 0.0))
                    done = float(bool(tr.get("done", False)))

                    model.replay_buffer.add(
                        np.expand_dims(obs, axis=0),
                        np.expand_dims(next_obs, axis=0),
                        np.expand_dims(action, axis=0),
                        np.array([reward], dtype=np.float32),
                        np.array([done], dtype=np.float32),
                        [info],
                    )
                    inserted += 1
                except Exception:
                    continue

    print(
        f"[train_SAC] replay preload inserted transitions: {inserted}, "
        f"reward_recompute={'on' if (reward_cfg or {}) else 'off'}, fail_keep_k={int(consecutive_fail_keep_k)}"
    )


def train_sac(go2_cfg, terrain_cfg,
              total_timesteps=20000,
              max_episode_steps=35,
              log_dir="train_terrain_logs",
              device="auto",
              learning_rate=3e-4,
              batch_size=256,
              buffer_size=1_000_000,
              learning_starts=100,
              train_freq=1,
              gradient_steps=1,
              tau=0.005,
              gamma=0.99,
              seed=0,
              plot_save_every_steps=2000,
              plot_smooth_window=20,
              checkpoint_every_steps=10000,
              preload_model_path="",
              preload_pkl_paths=None,
              failure_pkl_name="train_failure_chains.pkl",
              failure_flush_every_episodes=50,
              reward_cfg=None,
              consecutive_fail_keep_k: int = 0,
              classifier_gate=None,
              classifier_threshold=0.5,
              min_rl_buffer_size_for_update=1000):

    preload_pkl_paths = preload_pkl_paths or []

    def make_env():
        trainer = TerrainTrainer(go2_cfg, terrain_cfg)
        env = TerrainGymEnv(trainer, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env = FailureRecordingWrapper(
            env,
            out_dir=log_dir,
            pkl_name=failure_pkl_name,
            flush_every_episodes=failure_flush_every_episodes,
        )
        return env

    vec_env = DummyVecEnv([make_env])

    action_dim = int(np.prod(vec_env.action_space.shape))
    if action_dim <= 0:
        raise ValueError(
            "Terrain action dimension is 0, SAC cannot train with empty action space. "
            "Please check terrain config and ensure terrain_action.terrain_types includes at least one controllable type "
            "(e.g. ['bump'])."
        )

    gated_mode = classifier_gate is not None

    model_cls = GatedSAC if gated_mode else SAC
    model_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device=device,
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        buffer_size=int(buffer_size),
        learning_starts=int(learning_starts),
        train_freq=int(train_freq),
        gradient_steps=int(gradient_steps),
        tau=float(tau),
        gamma=float(gamma),
        seed=int(seed),
        replay_buffer_class=FilteredGatedReplayBuffer,
        replay_buffer_kwargs={"consecutive_fail_keep_k": int(consecutive_fail_keep_k)},
    )
    if gated_mode:
        model_kwargs.update(
            gate=classifier_gate,
            gate_threshold=float(classifier_threshold),
            min_rl_buffer_size_for_update=int(min_rl_buffer_size_for_update),
            consecutive_fail_keep_k=int(consecutive_fail_keep_k),
        )
    model = model_cls(**model_kwargs)

    if preload_model_path and os.path.exists(preload_model_path):
        print(f"[train_SAC] warm-loading pretrained weights from: {preload_model_path}")
        model.set_parameters(preload_model_path, exact_match=False, device=device)

    if preload_pkl_paths:
        preload_replay_buffer_from_pkl(
            model,
            preload_pkl_paths,
            reward_cfg=reward_cfg,
            consecutive_fail_keep_k=int(consecutive_fail_keep_k),
        )

    callback = TrainingLoggerCallback(
        out_dir=log_dir,
        save_every_steps=int(plot_save_every_steps),
        smooth_window=int(plot_smooth_window),
        checkpoint_every_steps=int(checkpoint_every_steps),
    )

    model.learn(total_timesteps=total_timesteps,
                callback=callback)

    if classifier_gate:
        print(
            f"[train_SAC] gated action stats: rl={model.rl_action_count}, random={model.random_action_count}, "
            f"threshold={classifier_threshold}"
        )

    model.save(os.path.join(log_dir, "model.zip"))


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    train_config_file = "train_config.yaml"
    with open(f"{current_path}/{train_config_file}", "r", encoding="utf-8") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    configure_torch_runtime(train_config)

    log_dir = f"{current_path}/train_logs/{train_config['log_name']}"
    os.makedirs(log_dir, exist_ok=True)
    terrain_cfg_file = os.path.join(current_path, train_config["terrain_config"])
    reward_cfg = _extract_reward_cfg_from_terrain_yaml(terrain_cfg_file)
    train_cfg_file = os.path.join(current_path, train_config_file)
    os.system(f"cp {terrain_cfg_file} {log_dir}")
    os.system(f"cp {train_cfg_file} {log_dir}")

    go2_cfg = [train_config['go2_task'], train_config['go2_config']]
    terrain_cfg = f"train_SAC/{train_config['terrain_config']}"
    total_timesteps = train_config['total_timesteps']
    max_episode_steps = train_config['max_episode_steps']

    preload_pkl_paths = train_config.get("preload_pkl_paths", [])
    for i in range(len(preload_pkl_paths)):
        preload_pkl_paths[i] = os.path.join(current_path, preload_pkl_paths[i])

    classifier_gate = None
    gate_cfg = train_config.get("classifier_gate", {})
    if bool(gate_cfg.get("enabled", False)):
        ckpt = str(gate_cfg.get("checkpoint_path", "")).strip()
        if not ckpt:
            raise ValueError("classifier_gate.enabled=true but checkpoint_path is empty")
        if not os.path.isabs(ckpt):
            ckpt = os.path.normpath(os.path.join(current_path, ckpt))
        classifier_gate = ClassifierGate(
            checkpoint_path=ckpt,
            device=str(gate_cfg.get("device", train_config.get("device", "cpu"))),
            hidden_dim=int(gate_cfg.get("hidden_dim", 1024)),
            concat_action_to_obs=bool(gate_cfg.get("concat_action_to_obs", True)),
        )

    train_sac(
        go2_cfg,
        terrain_cfg,
        total_timesteps=total_timesteps,
        max_episode_steps=max_episode_steps,
        log_dir=log_dir,
        device=train_config.get("device", "auto"),
        learning_rate=train_config.get("learning_rate", 3e-4),
        batch_size=train_config.get("batch_size", 256),
        buffer_size=train_config.get("buffer_size", 1_000_000),
        learning_starts=train_config.get("learning_starts", 100),
        train_freq=train_config.get("train_freq", 1),
        gradient_steps=train_config.get("gradient_steps", 1),
        tau=train_config.get("tau", 0.005),
        gamma=train_config.get("gamma", 0.99),
        seed=train_config.get("seed", 0),
        plot_save_every_steps=train_config.get("plot_save_every_steps", 2000),
        plot_smooth_window=train_config.get("plot_smooth_window", 20),
        checkpoint_every_steps=train_config.get("checkpoint_every_steps", 10000),
        preload_model_path=train_config.get("preload_model_path", ""),
        preload_pkl_paths=preload_pkl_paths,
        failure_pkl_name=train_config.get("failure_pkl_name", "train_failure_chains.pkl"),
        failure_flush_every_episodes=train_config.get("failure_flush_every_episodes", 50),
        reward_cfg=reward_cfg,
        consecutive_fail_keep_k=int(train_config.get("consecutive_fail_keep_k", 0)),
        classifier_gate=classifier_gate,
        classifier_threshold=float(gate_cfg.get("threshold", 0.5)),
        min_rl_buffer_size_for_update=int(gate_cfg.get("min_rl_buffer_size_for_update", train_config.get("learning_starts", 100))),
    )


if __name__ == "__main__":
    main()
