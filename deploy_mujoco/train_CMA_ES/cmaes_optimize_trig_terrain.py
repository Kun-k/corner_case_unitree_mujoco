import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import json
import math
from dataclasses import dataclass

import numpy as np
import yaml

from deploy_mujoco.terrain_trainer import TerrainTrainer


@dataclass
class CMAESConfig:
    dim: int
    sigma: float = 0.8
    popsize: int = 16
    seed: int = 0


class SimpleCMAES:
    """Minimal CMA-ES implementation (ask/tell) for continuous optimization."""

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
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff),
        )
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
        # CMA-ES assumes minimization.
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

        # Numerical symmetrization + eigendecomposition.
        self.C = 0.5 * (self.C + self.C.T)
        d, b = np.linalg.eigh(self.C)
        d = np.maximum(d, 1e-20)
        self.D = np.sqrt(d)
        self.B = b
        self.inv_sqrt_C = np.dot(self.B, np.dot(np.diag(1.0 / self.D), self.B.T))

        self.generation += 1


def decode_params_to_angle_array(x, mode_y, mode_x, terrain_amplitude_max=1.0):
    """Map CMA vector to angle_array[my,mx,2] = [theta, amplitude]."""
    n = mode_y * mode_x
    theta = np.reshape(x[:n], (mode_y, mode_x))
    alt_raw = np.reshape(x[n : 2 * n], (mode_y, mode_x))

    theta = np.mod(theta, 2.0 * np.pi)
    alt = np.tanh(alt_raw) * float(terrain_amplitude_max)
    angle_array = np.stack([theta, alt], axis=-1)
    return angle_array.astype(np.float32)


def evaluate_candidate(trainer, angle_array, max_robot_steps, safe_radius_m, blend_radius_m):
    # Reset robot + sim first.
    trainer.reset()

    # Build terrain from trig params and enforce safe spawn around (0,0).
    trainer.terrain_changer.generate_trig_terrain(angle_array)
    trainer.terrain_changer.enforce_safe_spawn_area(
        center_world=(0.0, 0.0),
        safe_radius_m=safe_radius_m,
        blend_radius_m=blend_radius_m,
        target_height=0.0,
    )

    if trainer.render:
        trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
        trainer.viewer.sync()

    total_reward = 0.0
    done = False

    for _ in range(max_robot_steps):
        _, _, reward, done, _ = trainer.step_only_robot()

        total_reward += float(reward)
        if done:
            break

    # maximize reward -> CMA-ES minimization fitness
    fitness = -total_reward
    return fitness, total_reward, done


def main():
    current_path = os.path.dirname(os.path.realpath(__file__))

    train_config_file = "train_config.yaml"
    with open(f"{current_path}/{train_config_file}", "r", encoding="utf-8") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    log_dir = f"{current_path}/logs/{train_config['log_name']}"
    os.makedirs(log_dir, exist_ok=True)
    terrain_cfg_file = os.path.join(current_path, train_config["terrain_config"])
    train_cfg_file = os.path.join(current_path, train_config_file)
    os.system(f"cp {terrain_cfg_file} {log_dir}")
    os.system(f"cp {train_cfg_file} {log_dir}")

    trainer = TerrainTrainer([train_config['go2_task'], train_config['go2_config']], f"train_CMA_ES/{train_config['terrain_config']}")

    trainer.init_skip_time = 0
    trainer.init_skip_frame = 10

    dim = 2 * train_config['mode_y'] * train_config['mode_x']
    cma = SimpleCMAES(CMAESConfig(dim=dim, sigma=train_config['sigma'], popsize=train_config['popsize'], seed=train_config['seed']))

    best_fitness = float("inf")
    best_reward = -float("inf")
    best_x = None
    history = []

    try:
        for gen in range(train_config['generations']):
            xs, zs = cma.ask()

            fit = np.zeros(train_config['popsize'], dtype=np.float64)
            rew = np.zeros(train_config['popsize'], dtype=np.float64)
            done_rate = 0

            for i in range(train_config['popsize']):
                angle_array = decode_params_to_angle_array(
                    xs[i],
                    train_config['mode_y'],
                    train_config['mode_x'],
                    train_config.get('terrain_amplitude_max', 1.0),
                )
                f_i, r_i, done_i = evaluate_candidate(
                    trainer,
                    angle_array,
                    max_robot_steps=train_config['max_robot_steps'],
                    safe_radius_m=train_config['safe_radius_m'],
                    blend_radius_m=train_config['blend_radius_m'],
                )
                fit[i] = f_i
                rew[i] = r_i
                done_rate += int(done_i)

            cma.tell(xs, zs, fit)

            gen_best_idx = int(np.argmin(fit))
            gen_best_fit = float(fit[gen_best_idx])
            gen_best_rew = float(rew[gen_best_idx])
            gen_mean_rew = float(np.mean(rew))
            gen_done_rate = float(done_rate / train_config['popsize'])

            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_reward = gen_best_rew
                best_x = xs[gen_best_idx].copy()

            row = {
                "generation": gen,
                "best_fitness": gen_best_fit,
                "best_reward": gen_best_rew,
                "mean_reward": gen_mean_rew,
                "done_rate": gen_done_rate,
                "sigma": float(cma.sigma),
            }
            history.append(row)
            print(
                f"[gen {gen:03d}] best_reward={gen_best_rew:.3f} mean_reward={gen_mean_rew:.3f} "
                f"done_rate={gen_done_rate:.2f} sigma={cma.sigma:.4f}"
            )

        if best_x is not None:
            best_angle_array = decode_params_to_angle_array(
                best_x,
                train_config['mode_y'],
                train_config['mode_x'],
                train_config.get('terrain_amplitude_max', 1.0),
            )
            np.save(os.path.join(log_dir, "best_angle_array.npy"), best_angle_array)
            np.save(os.path.join(log_dir, "best_vector.npy"), best_x)

        with open(os.path.join(log_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        summary = {
            "best_fitness": best_fitness,
            "best_reward": best_reward,
            "generations": train_config['generations'],
            "popsize": train_config['popsize'],
            "mode_y": train_config['mode_y'],
            "mode_x": train_config['mode_x'],
            "safe_radius_m": train_config['safe_radius_m'],
            "blend_radius_m": train_config['blend_radius_m'],
            "terrain_amplitude_max": float(train_config.get('terrain_amplitude_max', 1.0)),
        }
        with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Done. best_reward={best_reward:.3f}, logs in: {log_dir}")

    finally:
        trainer.close_viewer()


if __name__ == "__main__":
    main()


'''
nohup python deploy_mujoco/train_CMA_ES/cmaes_optimize_trig_terrain.py --go2-task terrain --go2-config go2.yaml --terrain-config terrain_config.yaml --generations 100 --popsize 8 --mode-y 10 --mode-x 10  >emaes.out 2>&1 &
'''
