import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import yaml

from deploy.deploy_mujoco_go2.offline_data_utils import load_chains_from_pkl_file


def _resolve_path(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def load_logs_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("logs_config must be a dict yaml")
    return cfg


def get_log_dirs(logs_config_path: str) -> Tuple[List[str], str]:
    cfg = load_logs_config(logs_config_path)
    base_dir = os.path.dirname(os.path.abspath(logs_config_path))
    log_dirs_cfg = cfg.get("log_dirs", [])
    if not isinstance(log_dirs_cfg, list):
        raise ValueError("log_dirs must be a list")
    log_dirs = [_resolve_path(base_dir, p) for p in log_dirs_cfg]
    output_dir = _resolve_path(base_dir, cfg.get("output_dir", "./train_logs/offline_stats"))
    return log_dirs, output_dir


def get_log_loading_options(logs_config_path: str) -> Dict:
    cfg = load_logs_config(logs_config_path)
    return {
        "consecutive_fail_keep_k": int(cfg.get("consecutive_fail_keep_k", 0)),
    }


def _build_states_output_dir_from_train_cfg(train_cfg: Dict, base_dir: str) -> str:
    states_cfg = train_cfg.get("states_logs", {}) if isinstance(train_cfg, dict) else {}
    if isinstance(states_cfg, dict) and states_cfg.get("output_dir", None):
        return _resolve_path(base_dir, str(states_cfg["output_dir"]))

    log_name = "offline_stats"
    if isinstance(states_cfg, dict) and states_cfg.get("log_name", None):
        log_name = str(states_cfg["log_name"])
    elif isinstance(train_cfg, dict) and train_cfg.get("log_name", None):
        log_name = str(train_cfg["log_name"])
    return os.path.join(base_dir, "states_logs", log_name)


def get_log_dirs_and_output_from_train_cfg(train_cfg: Dict, base_dir: str) -> Tuple[List[str], str]:
    """Resolve log_dirs/output_dir from merged train_config, fallback to legacy logs_config.yaml."""
    log_dirs_cfg = train_cfg.get("log_dirs", []) if isinstance(train_cfg, dict) else []
    log_dirs = [_resolve_path(base_dir, str(p)) for p in log_dirs_cfg]
    output_dir = _build_states_output_dir_from_train_cfg(train_cfg, base_dir)
    return log_dirs, output_dir


def get_log_loading_options_from_train_cfg(train_cfg: Dict, base_dir: str) -> Dict:
    return {
        "consecutive_fail_keep_k": int(train_cfg.get("consecutive_fail_keep_k", 0)),
    }


def load_transition_chains_from_logs(log_dirs: List[str], consecutive_fail_keep_k: int = 0) -> List[List[Dict]]:
    """Load filtered transition chains (episode-wise) from configured log dirs."""
    chains_out: List[List[Dict]] = []
    for d in log_dirs:
        pkl_path = os.path.join(d, "collision_failures.pkl")
        if not os.path.exists(pkl_path):
            continue
        try:
            chains = load_chains_from_pkl_file(
                pkl_path,
                consecutive_fail_keep_k=int(consecutive_fail_keep_k),
            )
            chains_out.extend(chains)
        except Exception:
            continue
    return chains_out


def load_transitions_from_logs(log_dirs: List[str]) -> List[Dict]:
    transitions: List[Dict] = []
    for chain in load_transition_chains_from_logs(log_dirs):
        for tr in chain:
            if isinstance(tr, dict) and "obs" in tr and "action" in tr:
                transitions.append(tr)
    return transitions


def _read_last_csv_row(csv_path: str) -> Dict:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        return {}
    return rows[-1]


def aggregate_failure_summary_csv(log_dirs: List[str]) -> Dict[str, float]:
    keys = [
        "episodes_evaluated",
        "total_failures",
        "collision_failures",
        "fall_failures",
        "base_collision_failures",
        "thigh_collision_failures",
        "stuck_failures",
    ]
    agg = {k: 0.0 for k in keys}

    for d in log_dirs:
        csv_path = os.path.join(d, "failure_summary.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            row = _read_last_csv_row(csv_path)
        except Exception:
            continue
        for k in keys:
            if k in row and row[k] != "":
                try:
                    agg[k] += float(row[k])
                except Exception:
                    pass

    total = max(agg["episodes_evaluated"], 1.0)
    for k in keys:
        if k == "episodes_evaluated":
            continue
        agg[f"prob_{k}"] = float(agg[k] / total)

    return agg


def stack_state_action(transitions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    states = []
    actions = []
    for tr in transitions:
        s = np.asarray(tr.get("obs", []), dtype=np.float32)
        a = np.asarray(tr.get("action", []), dtype=np.float32)
        if s.size == 0 or a.size == 0:
            continue
        states.append(s)
        actions.append(a)
    if len(states) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    return np.stack(states, axis=0), np.stack(actions, axis=0)


def build_transition_arrays(transitions: List[Dict]):
    states = []
    actions = []
    dones = []
    infos = []
    for tr in transitions:
        s = np.asarray(tr.get("obs", []), dtype=np.float32)
        a = np.asarray(tr.get("action", []), dtype=np.float32)
        if s.size == 0 or a.size == 0:
            continue
        states.append(s)
        actions.append(a)
        dones.append(bool(tr.get("done", False)))
        infos.append(tr.get("info", {}))

    if len(states) == 0:
        return {
            "states": np.zeros((0, 0), dtype=np.float32),
            "actions": np.zeros((0, 0), dtype=np.float32),
            "dones": np.zeros((0,), dtype=np.bool_),
            "infos": [],
        }

    return {
        "states": np.stack(states, axis=0),
        "actions": np.stack(actions, axis=0),
        "dones": np.asarray(dones, dtype=np.bool_),
        "infos": infos,
    }
