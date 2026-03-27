import csv
import os
import pickle
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from deploy.deploy_mujoco_go2.train_offline.data_io import get_log_dirs_and_output_from_train_cfg
from deploy.deploy_mujoco_go2.reward_recompute_utils import (
    load_reward_cfg_from_yaml,
    recompute_fail_flags_from_info,
)


RHW_KEYS = [
    "RHW",
    "rhw",
    "r_h_w",
    "reward_hard_weight",
    "reward_hardness_weight",
]


def _read_last_csv_row(csv_path: str) -> dict:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        return {}
    return rows[-1]


def _read_total_episodes_from_csv(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    try:
        row = _read_last_csv_row(csv_path)
        return int(float(row.get("episodes_evaluated", 0.0)))
    except Exception:
        return 0


def _safe_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _extract_rhw_from_row(row: dict):
    for k in RHW_KEYS:
        if k in row:
            v = _safe_float(row.get(k))
            if v is not None:
                return v
    return None


def _load_local_rhw_from_csv(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        return {}
    ep_to_rhw = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep = _safe_float(row.get("episodes_evaluated", None))
                rhw = _extract_rhw_from_row(row)
                if ep is None or rhw is None:
                    continue
                ep_i = int(ep)
                if ep_i >= 1:
                    ep_to_rhw[ep_i] = rhw
    except Exception:
        return {}
    return ep_to_rhw


def _episode_has_effective_failure(chain: list, reward_cfg: dict) -> bool:
    if not isinstance(chain, list):
        return False
    for tr in chain:
        info = tr.get("info", {}) if isinstance(tr, dict) else {}
        if bool(recompute_fail_flags_from_info(info, reward_cfg).get("any_fail", False)):
            return True
    return False


def _extract_failure_episodes_from_obj(obj, reward_cfg: dict) -> List[int]:
    episodes: List[int] = []

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "episode" in item:
                if isinstance(item.get("chain", None), list):
                    if not _episode_has_effective_failure(item.get("chain", []), reward_cfg):
                        continue
                try:
                    episodes.append(int(item["episode"]))
                except Exception:
                    continue
    elif isinstance(obj, dict):
        if "episode" in obj:
            try:
                episodes.append(int(obj["episode"]))
            except Exception:
                pass
        if "episodes" in obj and isinstance(obj["episodes"], list):
            for v in obj["episodes"]:
                try:
                    episodes.append(int(v))
                except Exception:
                    continue

    return episodes


def _load_failure_episodes_from_pkl(pkl_path: str, reward_cfg: dict) -> List[int]:
    if not os.path.exists(pkl_path):
        return []
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return []
    return _extract_failure_episodes_from_obj(obj, reward_cfg)


def _extract_rhw_from_transition(tr: dict):
    if not isinstance(tr, dict):
        return None

    for k in RHW_KEYS:
        if k in tr:
            v = _safe_float(tr.get(k))
            if v is not None:
                return v

    info = tr.get("info", {})
    if isinstance(info, dict):
        for k in RHW_KEYS:
            if k in info:
                v = _safe_float(info.get(k))
                if v is not None:
                    return v
    return None


def _extract_transition_is_weight(tr: dict):
    if not isinstance(tr, dict):
        return None

    # Prefer pre-computed importance weight if provided.
    for key in ["importance_weight", "is_weight"]:
        v = _safe_float(tr.get(key, None))
        if v is not None and v > 0.0:
            return float(v)

    info = tr.get("info", {}) if isinstance(tr.get("info", {}), dict) else {}
    for key in ["importance_weight", "is_weight"]:
        v = _safe_float(info.get(key, None))
        if v is not None and v > 0.0:
            return float(v)

    # Otherwise derive IS weight from recorded behavior density.
    # w = p_random(a|s) / p_data(a|s), where p_random is uniform on [-1, 1]^dim.
    data_density = _safe_float(tr.get("action_prob_density", None))
    if data_density is None:
        data_density = _safe_float(info.get("action_prob_density", None))
    if data_density is None or data_density <= 0.0:
        return None

    action = np.asarray(tr.get("action", []), dtype=np.float64).reshape(-1)
    if action.size <= 0:
        return None

    random_density = float((0.5) ** int(action.size))
    return float(random_density / float(data_density))


def _extract_episode_rhw_from_obj(obj) -> dict:
    ep_to_rhw = {}

    # Expected eval format: list of {"episode": int, "chain": [transition, ...]}
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict) or "episode" not in item:
                continue
            ep_raw = _safe_float(item.get("episode"))
            if ep_raw is None:
                continue
            chain = item.get("chain", [])
            if not isinstance(chain, list):
                continue

            vals = []
            for tr in chain:
                v = _extract_rhw_from_transition(tr)
                if v is not None:
                    vals.append(v)

            if len(vals) > 0:
                ep_to_rhw[int(ep_raw)] = float(np.mean(np.asarray(vals, dtype=np.float64)))

    return ep_to_rhw


def _extract_episode_isw_from_obj(obj) -> dict:
    ep_to_isw = {}

    if not isinstance(obj, list):
        return ep_to_isw

    for item in obj:
        if not isinstance(item, dict) or "episode" not in item:
            continue
        ep_raw = _safe_float(item.get("episode"))
        if ep_raw is None:
            continue
        chain = item.get("chain", [])
        if not isinstance(chain, list):
            continue

        log_w = 0.0
        weight = 1
        weight_sum = 0.0
        valid = False

        for t, tr in enumerate(chain):
            # if t < int(0.9 * len(chain)):
            #     continue
            w = _extract_transition_is_weight(tr)
            if w is not None and np.isfinite(w) and w > 0.0:
                log_w += np.log(w)
                weight *= w
                weight_sum += w
                valid = True

        if valid:
            # log_w = np.clip(log_w, -10.0, 10.0)
            ep_to_isw[int(ep_raw)] = float(np.exp(log_w))
            # ep_to_isw[int(ep_raw)] = weight
            # ep_to_isw[int(ep_raw)] = weight_sum / len(chain)
        else:
            ep_to_isw[int(ep_raw)] = 1.0

    return ep_to_isw


def _load_episode_rhw_from_pkl(pkl_path: str) -> dict:
    if not os.path.exists(pkl_path):
        return {}
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return {}
    return _extract_episode_rhw_from_obj(obj)


def _load_episode_isw_from_pkl(pkl_path: str) -> dict:
    if not os.path.exists(pkl_path):
        return {}
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return {}
    return _extract_episode_isw_from_obj(obj)


def _normalize_episode_base(fail_episodes: List[int], total_episodes: int) -> np.ndarray:
    if len(fail_episodes) == 0:
        return np.zeros((0,), dtype=np.int64)

    arr = np.asarray(fail_episodes, dtype=np.int64)
    if total_episodes <= 0:
        return arr

    # Eval logger currently stores episode as 0-based index.
    # If data appears 1-based (all in [1, total_episodes]), keep as-is.
    if arr.min() >= 1 and arr.max() <= total_episodes:
        return arr

    # Otherwise treat as 0-based and convert to 1-based episode count axis.
    return arr + 1


def _build_global_curve(log_dirs: List[str], reward_cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episode_offset = 0
    global_failure_episodes: List[int] = []
    global_rhw_points = {}

    for d in log_dirs:
        csv_path = os.path.join(d, "failure_summary.csv")
        pkl_path = os.path.join(d, "collision_failures.pkl")

        total_episodes = _read_total_episodes_from_csv(csv_path)
        local_fail_eps = _load_failure_episodes_from_pkl(pkl_path, reward_cfg=reward_cfg)
        local_fail_eps = _normalize_episode_base(local_fail_eps, total_episodes)

        local_rhw_csv = _load_local_rhw_from_csv(csv_path)
        local_rhw_pkl = _load_episode_rhw_from_pkl(pkl_path)

        if total_episodes > 0:
            valid = local_fail_eps[(local_fail_eps >= 1) & (local_fail_eps <= total_episodes)]
        else:
            valid = local_fail_eps[local_fail_eps >= 1]

        for ep in valid.tolist():
            global_failure_episodes.append(int(ep + episode_offset))

        # Normalize and merge RHW sources: prefer CSV value at the same episode index.
        for ep_raw, rhw in local_rhw_pkl.items():
            ep_local = int(ep_raw)
            if total_episodes > 0 and (ep_local < 1 or ep_local > total_episodes):
                ep_local = ep_local + 1
            if ep_local >= 1:
                local_rhw_csv.setdefault(ep_local, rhw)

        for ep_local, rhw in local_rhw_csv.items():
            if total_episodes > 0 and (ep_local < 1 or ep_local > total_episodes):
                continue
            ep_global = int(ep_local + episode_offset)
            global_rhw_points[ep_global] = float(rhw)

        episode_offset += max(total_episodes, 0)

    total_global_episodes = int(episode_offset)
    if total_global_episodes <= 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    episodes = np.arange(0, total_global_episodes + 1, dtype=np.float64)
    fail_flags = np.zeros((total_global_episodes + 1,), dtype=np.float64)
    is_weights = np.ones((total_global_episodes + 1,), dtype=np.float64)
    is_weights[0] = 0.0

    has_any_is_weight = False

    # Second pass: attach episode-level IS weights from PKL if available.
    episode_offset = 0
    for d in log_dirs:
        csv_path = os.path.join(d, "failure_summary.csv")
        pkl_fail_path = os.path.join(d, "collision_failures.pkl")
        pkl_non_fail_path = os.path.join(d, "non_failure_trajectories.pkl")

        total_episodes = _read_total_episodes_from_csv(csv_path)
        local_isw = {}

        local_isw.update(_load_episode_isw_from_pkl(pkl_fail_path))
        local_isw.update(_load_episode_isw_from_pkl(pkl_non_fail_path))

        for ep_raw, w in local_isw.items():
            ep_local = int(ep_raw) + 1
            # if total_episodes > 0 and (ep_local < 1 or ep_local > total_episodes):
            #     ep_local = ep_local + 1
            # if total_episodes > 0 and (ep_local < 1 or ep_local > total_episodes):
            #     continue
            ep_global = int(ep_local + episode_offset)
            if 1 <= ep_global <= total_global_episodes and np.isfinite(w) and float(w) > 0.0:
                is_weights[ep_global] = float(w)
                has_any_is_weight = True

        episode_offset += max(total_episodes, 0)

    if len(global_failure_episodes) > 0:
        fail_ep = np.asarray(global_failure_episodes, dtype=np.int64)
        fail_ep = fail_ep[(fail_ep >= 1) & (fail_ep <= total_global_episodes)]
        if fail_ep.size > 0:
            uniq_fail_ep = np.unique(fail_ep)
            fail_flags[uniq_fail_ep] = 1.0

    weighted_fail_flags = fail_flags * is_weights
    cum_weighted_fail = np.cumsum(weighted_fail_flags)
    cum_weight = np.cumsum(is_weights)
    # fail_rate = np.divide(cum_weighted_fail, cum_weight)
    # fail_rate = np.divide(cum_weighted_fail, np.concatenate(([1], np.arange(1, total_global_episodes + 1))))
    # 不要直接除以 episode 数量，要除以权重的总和
    weighted_fail_sum = np.cumsum(fail_flags * is_weights)

    is_weights_tmp = is_weights.copy()
    for i in range(len(is_weights_tmp)):
        if is_weights_tmp[i] == 1.0:
            is_weights_tmp[i] = is_weights_tmp[i - 1]
    # is_weights_tmp[is_weights == 1.0] = 0.0
    total_weight_sum = np.cumsum(is_weights_tmp)
    # 得到 SNIS 估计的失效概率
    total_weight_sum[0] = 1
    fail_rate = weighted_fail_sum / total_weight_sum

    # Weighted Bernoulli variance with effective sample size.
    cum_w2 = np.cumsum(is_weights * is_weights)
    cum_w2[0] = 1.0
    n_eff = np.divide(cum_weight * cum_weight, cum_w2)
    variance = fail_rate * (1.0 - fail_rate) / np.maximum(n_eff, 1.0)
    std = np.sqrt(np.maximum(variance, 0.0))

    rhw = np.full((total_global_episodes + 1,), np.nan, dtype=np.float64)
    for ep_global, v in global_rhw_points.items():
        if 0 <= ep_global <= total_global_episodes:
            rhw[ep_global] = float(v)

    if has_any_is_weight:
        print("[plot_fail_rate_trend_IS] Detected recorded densities/weights, using importance-weighted fail rate.")
    else:
        print("[plot_fail_rate_trend_IS] No recorded densities/weights found, fallback to unweighted fail rate.")

    return episodes, fail_rate, std, variance, rhw


def _save_csv(
    path: str,
    episodes: np.ndarray,
    fail_rate: np.ndarray,
    std: np.ndarray,
    var_est: np.ndarray,
    rhw: np.ndarray,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episodes", "fail_rate", "std_for_shadow", "var_for_shadow", "bernoulli_var_est", "rhw"])
        for ep, fr, s, v_est, rhw_v in zip(episodes, fail_rate, std, var_est, rhw):
            rhw_out = "" if np.isnan(rhw_v) else float(rhw_v)
            writer.writerow([float(ep), float(fr), float(s), float(s * s), float(v_est), rhw_out])


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_cfg_path = os.path.join(base_dir, "train_config.yaml")

    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f) or {}
    log_dirs, output_dir = get_log_dirs_and_output_from_train_cfg(train_cfg, base_dir)
    terrain_cfg_path = os.path.join(base_dir, train_cfg.get("terrain_config", "terrain_config.yaml"))
    reward_cfg = load_reward_cfg_from_yaml(terrain_cfg_path)

    os.makedirs(output_dir, exist_ok=True)

    episodes, fail_rate_mean, std_shadow, mean_var_est, rhw_series = _build_global_curve(log_dirs, reward_cfg=reward_cfg)
    if episodes.size == 0:
        raise RuntimeError("No valid episodes found from configured log folders (CSV/PKL).")

    csv_path = os.path.join(output_dir, "fail_rate_trend.csv")
    _save_csv(csv_path, episodes, fail_rate_mean, std_shadow, mean_var_est, rhw_series)

    lower = np.clip(fail_rate_mean - std_shadow, 0.0, 1.0)
    upper = np.clip(fail_rate_mean + std_shadow, 0.0, 1.0)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(episodes, fail_rate_mean, linewidth=2.0, label="Fail rate")
    ax1.fill_between(episodes, lower, upper, alpha=0.25, label="Fail-rate variance (±1 std)")
    ax1.set_xlabel("Episodes evaluated")
    ax1.set_ylabel("Fail rate")
    ax1.set_title("Fail rate + RHW trend")
    # ax1.set_ylim(0.0, 0.0001)
    ax1.set_ylim(0.0, max(0.001, float(np.nanmax(upper)) * 1.2 if upper.size > 0 else 0.01))
    ax1.grid(True, alpha=0.3)
    print(float(np.nanmax(upper)))

    if np.any(np.isfinite(rhw_series)):
        ax2 = ax1.twinx()
        ax2.plot(episodes, rhw_series, color="tab:red", linewidth=1.5, alpha=0.8, label="RHW")
        ax2.set_ylabel("RHW")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    else:
        ax1.legend(loc="best")

    plt.tight_layout()

    fig_path = os.path.join(output_dir, "fail_rate_trend.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Processed log folders: {len(log_dirs)}")
    print(f"RHW points: {int(np.sum(np.isfinite(rhw_series)))}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()

