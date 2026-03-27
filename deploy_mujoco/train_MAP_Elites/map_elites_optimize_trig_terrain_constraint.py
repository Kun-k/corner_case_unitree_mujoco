import json
import os
from dataclasses import dataclass

import numpy as np
import yaml

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer


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

    def add(self, vector, score, descriptor, done):
        dx, dy = float(descriptor[0]), float(descriptor[1])
        ix, iy = self._bin_index(dx, dy)
        if score > self.fitness[ix, iy]:
            if not np.isfinite(self.fitness[ix, iy]):
                self.count += 1
            self.fitness[ix, iy] = score
            self.vectors[ix, iy] = vector
            self.desc[ix, iy] = np.array([dx, dy], dtype=np.float64)
            self.done[ix, iy] = bool(done)
            return True, (ix, iy)
        return False, (ix, iy)

    def random_elite(self):
        ids = np.argwhere(np.isfinite(self.fitness))
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
            "score": float(self.fitness[ix, iy]),
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


def decode_params_to_angle_array(x, mode_y, mode_x):
    n = mode_y * mode_x
    theta = np.reshape(x[:n], (mode_y, mode_x))
    alt_raw = np.reshape(x[n : 2 * n], (mode_y, mode_x))
    theta = np.mod(theta, 2.0 * np.pi)
    alt = np.tanh(alt_raw)  # no external amp max
    return np.stack([theta, alt], axis=-1).astype(np.float32)


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
    h = np.asarray(terrain_h, dtype=np.float32)
    gx = np.gradient(h, axis=1)
    gy = np.gradient(h, axis=0)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    return np.array([float(np.std(h)), float(np.mean(grad_mag))], dtype=np.float32)


def evaluate_candidate(trainer, angle_array, cfg):
    trainer.reset()
    trainer.terrain_changer.generate_trig_terrain(angle_array)
    trainer.terrain_changer.enforce_safe_spawn_area(
        center_world=(0.0, 0.0),
        safe_radius_m=cfg["safe_radius_m"],
        blend_radius_m=cfg["blend_radius_m"],
        target_height=0.0,
    )

    terrain_snapshot = np.array(trainer.terrain_changer.hfield, dtype=np.float32)
    descriptor = compute_descriptors(terrain_snapshot)

    total_reward = 0.0
    done = False
    for _ in range(cfg["max_robot_steps"]):
        _, _, reward, done, _ = trainer.step_only_robot()
        total_reward += float(reward)
        if done:
            break

    amp_mean = float(np.mean(np.abs(angle_array[..., 1])))
    fail_score = float(cfg["fail_weight"]) * (1.0 if done else 0.0)
    rew_score = float(cfg["reward_weight"]) * total_reward

    target_amp = float(cfg["target_amp_mean"])
    hinge = max(0.0, amp_mean - target_amp)
    amp_penalty = float(cfg["amp_weight"]) * amp_mean + float(cfg["amp_hinge_weight"]) * hinge

    score = fail_score + rew_score - amp_penalty
    return score, descriptor, done, amp_mean


def export_archive(archive, mode_y, mode_x, out_dir):
    np.save(os.path.join(out_dir, "archive_fitness.npy"), archive.fitness)
    np.save(os.path.join(out_dir, "archive_vectors.npy"), archive.vectors)
    np.save(os.path.join(out_dir, "archive_descriptors.npy"), archive.desc)
    np.save(os.path.join(out_dir, "archive_done.npy"), archive.done)
    best = archive.best()
    if best is not None:
        np.save(os.path.join(out_dir, "best_angle_array.npy"), decode_params_to_angle_array(best["vector"], mode_y, mode_x))
        np.save(os.path.join(out_dir, "best_vector.npy"), best["vector"])


def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    train_config_file = "train_config_constraint.yaml"
    with open(f"{current_path}/{train_config_file}", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    log_dir = f"{current_path}/logs/{cfg['log_name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.system(f"cp {os.path.join(current_path, cfg['terrain_config'])} {log_dir}")
    os.system(f"cp {os.path.join(current_path, train_config_file)} {log_dir}")

    rng = np.random.RandomState(cfg['seed'])
    trainer = TerrainTrainer([cfg['go2_task'], cfg['go2_config']], f"train_MAP_Elites/{cfg['terrain_config']}")
    trainer.init_skip_time = 0
    trainer.init_skip_frame = 10

    dim = 2 * cfg['mode_y'] * cfg['mode_x']
    archive = MAPElitesArchive(
        MAPElitesConfig(
            dim=dim,
            bins_x=cfg['bins_x'],
            bins_y=cfg['bins_y'],
            desc_min_x=cfg['desc_min_x'],
            desc_max_x=cfg['desc_max_x'],
            desc_min_y=cfg['desc_min_y'],
            desc_max_y=cfg['desc_max_y'],
            seed=cfg['seed'],
        )
    )

    history = []

    try:
        for it in range(cfg['iterations']):
            elite = archive.random_elite()
            use_random = (elite is None) or (rng.rand() < cfg['init_random_ratio'])

            if use_random:
                x = sample_random_vector(rng, cfg['mode_y'], cfg['mode_x'])
                parent_bin = None
            else:
                parent, parent_bin = elite
                x = mutate_vector(rng, parent, cfg['mode_y'], cfg['mode_x'], cfg['sigma_theta'], cfg['sigma_alt'])

            angle_array = decode_params_to_angle_array(x, cfg['mode_y'], cfg['mode_x'])
            score, descriptor, done, amp_mean = evaluate_candidate(trainer, angle_array, cfg)
            inserted, cell = archive.add(x, score, descriptor, done)

            row = {
                "iteration": it,
                "score": float(score),
                "descriptor": [float(descriptor[0]), float(descriptor[1])],
                "inserted": bool(inserted),
                "cell": [int(cell[0]), int(cell[1])],
                "parent_bin": parent_bin,
                "done": bool(done),
                "amp_mean": float(amp_mean),
                "coverage": archive.coverage(),
                "qd_score": archive.qd_score(),
            }
            history.append(row)

            print(
                f"[it {it:04d}] score={score:.3f} done={done} amp={amp_mean:.4f} "
                f"inserted={inserted} coverage={archive.coverage():.3f} qd={archive.qd_score():.3f}"
            )

            if (it + 1) % cfg['save_every'] == 0:
                export_archive(archive, cfg['mode_y'], cfg['mode_x'], log_dir)
                with open(os.path.join(log_dir, "history.json"), "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

        export_archive(archive, cfg['mode_y'], cfg['mode_x'], log_dir)

        summary = {
            "iterations": cfg['iterations'],
            "coverage": archive.coverage(),
            "qd_score": archive.qd_score(),
            "mode_y": cfg['mode_y'],
            "mode_x": cfg['mode_x'],
            "fail_weight": cfg['fail_weight'],
            "reward_weight": cfg['reward_weight'],
            "amp_weight": cfg['amp_weight'],
            "amp_hinge_weight": cfg['amp_hinge_weight'],
            "target_amp_mean": cfg['target_amp_mean'],
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

