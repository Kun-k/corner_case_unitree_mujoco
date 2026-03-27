import json
import os
from dataclasses import dataclass

import numpy as np

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer
import yaml


@dataclass
class MAPElitesConfig:
    dim: int
    bins_x: int
    bins_y: int
    desc_min_x: float
    desc_max_x: float
    desc_min_y: float
    desc_max_y: float
    seed: int = 0


class MAPElitesArchive:
    """2D MAP-Elites archive with maximization objective."""

    def __init__(self, cfg: MAPElitesConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

        self.fitness = np.full((cfg.bins_x, cfg.bins_y), -np.inf, dtype=np.float64)
        self.vectors = np.full((cfg.bins_x, cfg.bins_y, cfg.dim), np.nan, dtype=np.float64)
        self.desc = np.full((cfg.bins_x, cfg.bins_y, 2), np.nan, dtype=np.float64)
        self.done = np.zeros((cfg.bins_x, cfg.bins_y), dtype=np.bool_)
        self.count = 0

    def _bin_index(self, dx, dy):
        x = (dx - self.cfg.desc_min_x) / (self.cfg.desc_max_x - self.cfg.desc_min_x + 1e-12)
        y = (dy - self.cfg.desc_min_y) / (self.cfg.desc_max_y - self.cfg.desc_min_y + 1e-12)
        ix = int(np.clip(np.floor(x * self.cfg.bins_x), 0, self.cfg.bins_x - 1))
        iy = int(np.clip(np.floor(y * self.cfg.bins_y), 0, self.cfg.bins_y - 1))
        return ix, iy

    def add(self, vector, reward, descriptor, done):
        dx, dy = float(descriptor[0]), float(descriptor[1])
        ix, iy = self._bin_index(dx, dy)
        if reward > self.fitness[ix, iy]:
            if not np.isfinite(self.fitness[ix, iy]):
                self.count += 1
            self.fitness[ix, iy] = reward
            self.vectors[ix, iy] = vector
            self.desc[ix, iy] = np.array([dx, dy], dtype=np.float64)
            self.done[ix, iy] = bool(done)
            return True, (ix, iy)
        return False, (ix, iy)

    def random_elite(self):
        mask = np.isfinite(self.fitness)
        ids = np.argwhere(mask)
        if len(ids) == 0:
            return None
        k = self.rng.randint(0, len(ids))
        ix, iy = ids[k]
        return self.vectors[ix, iy].copy(), (int(ix), int(iy))

    def best(self):
        if self.count == 0:
            return None
        idx = np.unravel_index(np.argmax(self.fitness), self.fitness.shape)
        ix, iy = int(idx[0]), int(idx[1])
        return {
            "idx": [ix, iy],
            "reward": float(self.fitness[ix, iy]),
            "vector": self.vectors[ix, iy].copy(),
            "descriptor": self.desc[ix, iy].copy(),
            "done": bool(self.done[ix, iy]),
        }

    def coverage(self):
        return float(self.count) / float(self.cfg.bins_x * self.cfg.bins_y)

    def qd_score(self):
        mask = np.isfinite(self.fitness)
        if not np.any(mask):
            return 0.0
        return float(np.sum(self.fitness[mask]))


def decode_params_to_angle_array(x, mode_y, mode_x, terrain_amplitude_max=1.0):
    """Map vector -> angle_array[my,mx,2] = [theta, amplitude_weight]."""
    n = mode_y * mode_x
    theta = np.reshape(x[:n], (mode_y, mode_x))
    alt_raw = np.reshape(x[n : 2 * n], (mode_y, mode_x))

    theta = np.mod(theta, 2.0 * np.pi)
    alt = np.tanh(alt_raw) * float(terrain_amplitude_max)
    angle_array = np.stack([theta, alt], axis=-1)
    return angle_array.astype(np.float32)


def sample_random_vector(rng, mode_y, mode_x):
    n = mode_y * mode_x
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(n,))
    alt_raw = rng.normal(0.0, 1.0, size=(n,))
    return np.concatenate([theta, alt_raw], axis=0)


def mutate_vector(rng, parent, mode_y, mode_x, sigma_theta, sigma_alt):
    n = mode_y * mode_x
    child = parent.copy()
    child[:n] += rng.normal(0.0, sigma_theta, size=(n,))
    child[n:] += rng.normal(0.0, sigma_alt, size=(n,))
    child[:n] = np.mod(child[:n], 2.0 * np.pi)
    return child


def compute_descriptors(terrain_h):
    """Behavior descriptors from generated terrain (2D).

    d0: height std
    d1: mean gradient magnitude
    """
    h = np.asarray(terrain_h, dtype=np.float32)
    gx = np.gradient(h, axis=1)
    gy = np.gradient(h, axis=0)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    d0 = float(np.std(h))
    d1 = float(np.mean(grad_mag))
    return np.array([d0, d1], dtype=np.float32)


def evaluate_candidate(trainer, angle_array, max_robot_steps, safe_radius_m, blend_radius_m):
    trainer.reset()

    trainer.terrain_changer.generate_trig_terrain(angle_array)
    trainer.terrain_changer.enforce_safe_spawn_area(
        center_world=(0.0, 0.0),
        safe_radius_m=safe_radius_m,
        blend_radius_m=blend_radius_m,
        target_height=0.0,
    )

    terrain_snapshot = np.array(trainer.terrain_changer.hfield, dtype=np.float32)
    descriptor = compute_descriptors(terrain_snapshot)

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

    return total_reward, descriptor, done


def export_archive(archive, mode_y, mode_x, out_dir, terrain_amplitude_max=1.0):
    np.save(os.path.join(out_dir, "archive_fitness.npy"), archive.fitness)
    np.save(os.path.join(out_dir, "archive_vectors.npy"), archive.vectors)
    np.save(os.path.join(out_dir, "archive_descriptors.npy"), archive.desc)
    np.save(os.path.join(out_dir, "archive_done.npy"), archive.done)

    best = archive.best()
    if best is not None:
        best_angle_array = decode_params_to_angle_array(best["vector"], mode_y, mode_x, terrain_amplitude_max)
        np.save(os.path.join(out_dir, "best_angle_array.npy"), best_angle_array)
        np.save(os.path.join(out_dir, "best_vector.npy"), best["vector"])


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

    rng = np.random.RandomState(train_config['seed'])

    trainer = TerrainTrainer([train_config['go2_task'], train_config['go2_config']], f"train_MAP_Elites/{train_config['terrain_config']}")

    trainer.init_skip_time = 0
    trainer.init_skip_frame = 10

    dim = 2 * train_config['mode_y'] * train_config['mode_x']
    archive = MAPElitesArchive(
        MAPElitesConfig(
            dim=dim,
            bins_x=train_config['bins_x'],
            bins_y=train_config['bins_y'],
            desc_min_x=train_config['desc_min_x'],
            desc_max_x=train_config['desc_max_x'],
            desc_min_y=train_config['desc_min_y'],
            desc_max_y=train_config['desc_max_y'],
            seed=train_config['seed'],
        )
    )

    history = []

    try:
        for it in range(train_config['iterations']):
            elite = archive.random_elite()
            use_random = (elite is None) or (rng.rand() < train_config['init_random_ratio'])

            if use_random:
                x = sample_random_vector(rng, train_config['mode_y'], train_config['mode_x'])
                parent_bin = None
            else:
                parent, parent_bin = elite
                x = mutate_vector(rng, parent, train_config['mode_y'], train_config['mode_x'], train_config['sigma_theta'], train_config['sigma_alt'])

            angle_array = decode_params_to_angle_array(
                x,
                train_config['mode_y'],
                train_config['mode_x'],
                train_config.get('terrain_amplitude_max', 1.0),
            )
            reward, descriptor, done = evaluate_candidate(
                trainer,
                angle_array,
                max_robot_steps=train_config['max_robot_steps'],
                safe_radius_m=train_config['safe_radius_m'],
                blend_radius_m=train_config['blend_radius_m'],
            )

            inserted, cell = archive.add(x, reward, descriptor, done)

            best = archive.best()
            row = {
                "iteration": it,
                "reward": float(reward),
                "descriptor": [float(descriptor[0]), float(descriptor[1])],
                "inserted": bool(inserted),
                "cell": [int(cell[0]), int(cell[1])],
                "parent_bin": parent_bin,
                "coverage": archive.coverage(),
                "qd_score": archive.qd_score(),
                "best_reward": None if best is None else float(best["reward"]),
            }
            history.append(row)

            print(
                f"[it {it:04d}] reward={reward:.3f} desc=({descriptor[0]:.4f},{descriptor[1]:.4f}) "
                f"inserted={inserted} coverage={archive.coverage():.3f} qd={archive.qd_score():.3f}"
            )

            if (it + 1) % train_config['save_every'] == 0:
                export_archive(
                    archive,
                    train_config['mode_y'],
                    train_config['mode_x'],
                    log_dir,
                    terrain_amplitude_max=train_config.get('terrain_amplitude_max', 1.0),
                )
                with open(os.path.join(log_dir, "history.json"), "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

        export_archive(
            archive,
            train_config['mode_y'],
            train_config['mode_x'],
            log_dir,
            terrain_amplitude_max=train_config.get('terrain_amplitude_max', 1.0),
        )

        summary = {
            "iterations": train_config['iterations'],
            "coverage": archive.coverage(),
            "qd_score": archive.qd_score(),
            "mode_y": train_config['mode_y'],
            "mode_x": train_config['mode_x'],
            "bins_x": train_config['bins_x'],
            "bins_y": train_config['bins_y'],
            "terrain_amplitude_max": float(train_config.get('terrain_amplitude_max', 1.0)),
            # "best": archive.best(),
        }

        with open(os.path.join(log_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Done. coverage={archive.coverage():.3f}, qd_score={archive.qd_score():.3f}, logs={log_dir}")

    finally:
        trainer.close_viewer()


if __name__ == "__main__":
    main()


"""
nohup python deploy/deploy_mujoco_go2/train_MAP_Elites/map_elites_optimize_trig_terrain.py \
  --go2-task terrain \
  --go2-config go2.yaml \
  --terrain-config terrain_config.yaml \
  --iterations 1000 \
  --mode-y 10 \
  --mode-x 10   >emaes.out 2>&1 &
"""
