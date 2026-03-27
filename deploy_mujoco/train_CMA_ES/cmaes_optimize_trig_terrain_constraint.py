import json
import math
import os
from dataclasses import dataclass

import numpy as np
import yaml

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer


@dataclass
class CMAESConfig:
    dim: int
    sigma: float = 0.8
    popsize: int = 16
    seed: int = 0


class SimpleCMAES:
    def __init__(self, cfg: CMAESConfig):
        self.dim = cfg.dim
        self.sigma = float(cfg.sigma)
        self.popsize = int(cfg.popsize)
        self.rng = np.random.RandomState(int(cfg.seed))

        self.mu = self.popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / np.sum(w)
        self.mueff = (np.sum(self.weights) ** 2) / np.sum(self.weights ** 2)

        n = self.dim
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs

        self.mean = np.zeros(n, dtype=np.float64)
        self.C = np.eye(n, dtype=np.float64)
        self.p_c = np.zeros(n, dtype=np.float64)
        self.p_s = np.zeros(n, dtype=np.float64)
        self.B = np.eye(n, dtype=np.float64)
        self.D = np.ones(n, dtype=np.float64)
        self.inv_sqrt_C = np.eye(n, dtype=np.float64)
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.generation = 0

    def ask(self):
        arz = self.rng.randn(self.popsize, self.dim)
        ary = np.dot(arz, (self.B * self.D).T)
        arx = self.mean + self.sigma * ary
        return arx, arz

    def tell(self, arx, arz, fitness):
        idx = np.argsort(fitness)
        arx = arx[idx]
        arz = arz[idx]

        x_old = self.mean.copy()
        self.mean = np.dot(self.weights, arx[: self.mu])

        z_w = np.dot(self.weights, arz[: self.mu])
        y_w = np.dot(z_w, self.B * self.D)

        self.p_s = (1.0 - self.cs) * self.p_s + math.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * np.dot(self.inv_sqrt_C, y_w)
        p_s_norm = np.linalg.norm(self.p_s)

        h_sigma = float(
            p_s_norm / math.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.generation + 1)))
            < (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n
        )

        self.p_c = (1.0 - self.cc) * self.p_c + h_sigma * math.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y_w

        rank_one = np.outer(self.p_c, self.p_c)
        rank_mu = np.zeros_like(self.C)
        y_k = (arx[: self.mu] - x_old) / self.sigma
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(y_k[i], y_k[i])

        self.C = (
            (1.0 - self.c1 - self.cmu) * self.C
            + self.c1 * (rank_one + (1.0 - h_sigma) * self.cc * (2.0 - self.cc) * self.C)
            + self.cmu * rank_mu
        )

        self.sigma *= math.exp((self.cs / self.damps) * (p_s_norm / self.chi_n - 1.0))

        self.C = 0.5 * (self.C + self.C.T)
        d, b = np.linalg.eigh(self.C)
        d = np.maximum(d, 1e-20)
        self.D = np.sqrt(d)
        self.B = b
        self.inv_sqrt_C = np.dot(self.B, np.dot(np.diag(1.0 / self.D), self.B.T))
        self.generation += 1


def decode_params_to_angle_array(x, mode_y, mode_x):
    """No external amplitude scaling: amplitude is constrained in objective instead."""
    n = mode_y * mode_x
    theta = np.reshape(x[:n], (mode_y, mode_x))
    alt_raw = np.reshape(x[n : 2 * n], (mode_y, mode_x))

    theta = np.mod(theta, 2.0 * np.pi)
    alt = np.tanh(alt_raw)  # fixed in [-1, 1]
    angle_array = np.stack([theta, alt], axis=-1)
    return angle_array.astype(np.float32)


def evaluate_candidate(trainer, angle_array, cfg):
    trainer.reset()

    trainer.terrain_changer.generate_trig_terrain(angle_array)
    trainer.terrain_changer.enforce_safe_spawn_area(
        center_world=(0.0, 0.0),
        safe_radius_m=cfg["safe_radius_m"],
        blend_radius_m=cfg["blend_radius_m"],
        target_height=0.0,
    )

    if trainer.render:
        trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
        trainer.viewer.sync()

    total_reward = 0.0
    done = False
    for _ in range(cfg["max_robot_steps"]):
        _, _, reward, done, _ = trainer.step_only_robot()
        total_reward += float(reward)
        if done:
            break

    amp = angle_array[..., 1]
    amp_mean = float(np.mean(np.abs(amp)))

    # Score to maximize: failure first, then reward.
    fail_score = float(cfg["fail_weight"]) * (1.0 if done else 0.0)
    rew_score = float(cfg["reward_weight"]) * total_reward
    score = fail_score + rew_score

    # Constraint-style amplitude penalty.
    target_amp = float(cfg["target_amp_mean"])
    hinge = max(0.0, amp_mean - target_amp)
    amp_penalty = float(cfg["amp_weight"]) * amp_mean + float(cfg["amp_hinge_weight"]) * hinge

    fitness = -score + amp_penalty  # CMA-ES minimizes
    return fitness, total_reward, done, amp_mean, score, amp_penalty


def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    train_config_file = "train_config_constraint.yaml"
    with open(f"{current_path}/{train_config_file}", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    log_dir = f"{current_path}/logs/{cfg['log_name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.system(f"cp {os.path.join(current_path, cfg['terrain_config'])} {log_dir}")
    os.system(f"cp {os.path.join(current_path, train_config_file)} {log_dir}")

    trainer = TerrainTrainer([cfg['go2_task'], cfg['go2_config']], f"train_CMA_ES/{cfg['terrain_config']}")
    trainer.init_skip_time = 0
    trainer.init_skip_frame = 10

    dim = 2 * cfg['mode_y'] * cfg['mode_x']
    cma = SimpleCMAES(CMAESConfig(dim=dim, sigma=cfg['sigma'], popsize=cfg['popsize'], seed=cfg['seed']))

    best_fitness = float("inf")
    best_x = None
    history = []

    try:
        for gen in range(cfg['generations']):
            xs, zs = cma.ask()
            fit = np.zeros(cfg['popsize'], dtype=np.float64)
            done_cnt = 0
            amp_vals = []

            for i in range(cfg['popsize']):
                angle_array = decode_params_to_angle_array(xs[i], cfg['mode_y'], cfg['mode_x'])
                f_i, r_i, done_i, amp_i, score_i, penalty_i = evaluate_candidate(trainer, angle_array, cfg)
                fit[i] = f_i
                done_cnt += int(done_i)
                amp_vals.append(amp_i)

            cma.tell(xs, zs, fit)

            best_idx = int(np.argmin(fit))
            if float(fit[best_idx]) < best_fitness:
                best_fitness = float(fit[best_idx])
                best_x = xs[best_idx].copy()

            row = {
                "generation": gen,
                "best_fitness": float(np.min(fit)),
                "mean_fitness": float(np.mean(fit)),
                "done_rate": float(done_cnt / cfg['popsize']),
                "mean_amp": float(np.mean(amp_vals)),
                "sigma": float(cma.sigma),
            }
            history.append(row)
            print(
                f"[gen {gen:03d}] best_fit={row['best_fitness']:.4f} done_rate={row['done_rate']:.2f} "
                f"mean_amp={row['mean_amp']:.4f} sigma={row['sigma']:.4f}"
            )

        if best_x is not None:
            best_angle_array = decode_params_to_angle_array(best_x, cfg['mode_y'], cfg['mode_x'])
            np.save(os.path.join(log_dir, "best_angle_array.npy"), best_angle_array)
            np.save(os.path.join(log_dir, "best_vector.npy"), best_x)

        with open(os.path.join(log_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        summary = {
            "best_fitness": best_fitness,
            "generations": cfg['generations'],
            "popsize": cfg['popsize'],
            "mode_y": cfg['mode_y'],
            "mode_x": cfg['mode_x'],
            "fail_weight": cfg['fail_weight'],
            "reward_weight": cfg['reward_weight'],
            "amp_weight": cfg['amp_weight'],
            "amp_hinge_weight": cfg['amp_hinge_weight'],
            "target_amp_mean": cfg['target_amp_mean'],
        }
        with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Done. logs in: {log_dir}")
    finally:
        trainer.close_viewer()


if __name__ == "__main__":
    main()

