import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from deploy.deploy_mujoco_go2.train_offline.data_io import (
    get_log_dirs_and_output_from_train_cfg,
    get_log_loading_options_from_train_cfg,
    load_transition_chains_from_logs,
    stack_state_action,
)
from deploy.deploy_mujoco_go2.reward_recompute_utils import (
    load_reward_cfg_from_yaml,
    recompute_fail_flags_from_info,
)


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


class FailureClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def transition_label(tr: dict, reward_cfg: dict) -> float:
    info = tr.get("info", {})
    failure = bool(recompute_fail_flags_from_info(info, reward_cfg).get("any_fail", False))
    return 1.0 if failure else 0.0


def _save_curve(values, title, out_path, smooth_window=10):
    if len(values) == 0:
        return
    arr = np.asarray(values, dtype=np.float32)
    fig = plt.figure()
    plt.plot(arr, linewidth=1.0)
    if arr.size >= smooth_window:
        k = np.ones((smooth_window,), dtype=np.float32) / float(smooth_window)
        sm = np.convolve(arr, k, mode="valid")
        plt.plot(np.arange(arr.size - sm.size, arr.size), sm, linewidth=2.0)
    plt.title(title)
    plt.grid(True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_metrics_csv(path, losses, accs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        val_losses = losses[1] if isinstance(losses, tuple) else []
        train_losses = losses[0] if isinstance(losses, tuple) else losses
        val_accs = accs[1] if isinstance(accs, tuple) else []
        train_accs = accs[0] if isinstance(accs, tuple) else accs
        n_rows = max(len(train_losses), len(train_accs), len(val_losses), len(val_accs))
        for i in range(n_rows):
            writer.writerow(
                [
                    i + 1,
                    train_losses[i] if i < len(train_losses) else "",
                    train_accs[i] if i < len(train_accs) else "",
                    val_losses[i] if i < len(val_losses) else "",
                    val_accs[i] if i < len(val_accs) else "",
                ]
            )


def _stratified_split_indices(labels: np.ndarray, val_ratio: float, test_ratio: float, seed: int):
    if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("classifier val_split and test_split must be >=0 and sum to < 1.0")

    n = int(labels.shape[0])
    all_idx = np.arange(n, dtype=np.int64)
    if n < 5:
        return all_idx, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    test_idx = []

    for cls in [0.0, 1.0]:
        cls_idx = np.where(labels == cls)[0]
        cls_idx = cls_idx.astype(np.int64)
        if cls_idx.size == 0:
            continue
        rng.shuffle(cls_idx)

        n_cls = int(cls_idx.size)
        n_test = int(round(n_cls * test_ratio))
        n_val = int(round(n_cls * val_ratio))

        if n_cls >= 3 and test_ratio > 0.0 and n_test == 0:
            n_test = 1
        if n_cls >= 3 and val_ratio > 0.0 and n_val == 0:
            n_val = 1

        # Keep at least one sample in train for this class when possible.
        if n_test + n_val >= n_cls:
            overflow = n_test + n_val - (n_cls - 1)
            if overflow > 0:
                reduce_test = min(overflow, n_test)
                n_test -= reduce_test
                overflow -= reduce_test
            if overflow > 0:
                n_val = max(0, n_val - overflow)

        test_part = cls_idx[:n_test]
        val_part = cls_idx[n_test : n_test + n_val]
        train_part = cls_idx[n_test + n_val :]

        test_idx.append(test_part)
        val_idx.append(val_part)
        train_idx.append(train_part)

    train_idx = np.concatenate(train_idx) if len(train_idx) > 0 else np.zeros((0,), dtype=np.int64)
    val_idx = np.concatenate(val_idx) if len(val_idx) > 0 else np.zeros((0,), dtype=np.int64)
    test_idx = np.concatenate(test_idx) if len(test_idx) > 0 else np.zeros((0,), dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _eval_on_subset(model, criterion, x_t, y_t, indices):
    if indices.size == 0:
        return 0.0, 0.0
    with torch.no_grad():
        logits = model(x_t[indices])
        loss = criterion(logits, y_t[indices])
        pred = (torch.sigmoid(logits) >= 0.5).float()
        acc = (pred == y_t[indices]).float().mean().item()
    return float(loss.item()), float(acc)


def _confusion_on_subset(model, x_t, y_t, indices):
    if indices.size == 0:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    with torch.no_grad():
        logits = model(x_t[indices])
        pred = (torch.sigmoid(logits) >= 0.5).float()
        gt = y_t[indices]

    tp = int(((pred == 1.0) & (gt == 1.0)).sum().item())
    fp = int(((pred == 1.0) & (gt == 0.0)).sum().item())
    tn = int(((pred == 0.0) & (gt == 0.0)).sum().item())
    fn = int(((pred == 0.0) & (gt == 1.0)).sum().item())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _save_confusion_csv(path: str, train_cm: dict, val_cm: dict, test_cm: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "TP", "FP", "TN", "FN"])
        writer.writerow(["train", train_cm["tp"], train_cm["fp"], train_cm["tn"], train_cm["fn"]])
        writer.writerow(["val", val_cm["tp"], val_cm["fp"], val_cm["tn"], val_cm["fn"]])
        writer.writerow(["test", test_cm["tp"], test_cm["fp"], test_cm["tn"], test_cm["fn"]])


def _expand_fail_labels_in_chain(chain: list, pre_k: int, reward_cfg: dict) -> np.ndarray:
    """Mark failure frames and their preceding k frames as positive within a chain."""
    n = len(chain)
    labels = np.zeros((n,), dtype=np.float32)
    if n == 0:
        return labels

    base_fail = np.asarray([transition_label(tr, reward_cfg) for tr in chain], dtype=np.float32)
    fail_indices = np.where(base_fail > 0.5)[0]
    if fail_indices.size == 0:
        return labels

    labels[fail_indices] = 1.0
    k = int(max(0, pre_k))
    if k <= 0:
        return labels

    for idx in fail_indices:
        start = max(0, int(idx) - k)
        labels[start:int(idx) + 1] = 1.0
    return labels


def _flatten_chains_and_labels(chains: list, pre_k: int, reward_cfg: dict):
    transitions = []
    labels = []
    for chain in chains:
        if not isinstance(chain, list) or len(chain) == 0:
            continue
        chain_labels = _expand_fail_labels_in_chain(chain, pre_k, reward_cfg)
        for i, tr in enumerate(chain):
            if isinstance(tr, dict) and "obs" in tr and "action" in tr:
                transitions.append(tr)
                labels.append(float(chain_labels[i]))
    return transitions, np.asarray(labels, dtype=np.float32)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "train_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    configure_torch_runtime(cfg)

    classifier_cfg = cfg.get("classifier", {}) if isinstance(cfg, dict) else {}
    classifier_log_name = classifier_cfg.get("log_name", cfg.get("log_name", "offline_default"))
    log_dir = os.path.join(base_dir, "classifier", str(classifier_log_name))
    os.makedirs(log_dir, exist_ok=True)

    reward_cfg_path = os.path.join(base_dir, cfg.get("terrain_config", "terrain_config.yaml"))
    reward_cfg = load_reward_cfg_from_yaml(reward_cfg_path)
    log_dirs, _ = get_log_dirs_and_output_from_train_cfg(cfg, base_dir)
    loading_opts = get_log_loading_options_from_train_cfg(cfg, base_dir)
    fail_keep_k = int(loading_opts.get("consecutive_fail_keep_k", 0))
    chains = load_transition_chains_from_logs(log_dirs, consecutive_fail_keep_k=fail_keep_k)

    pre_k = int(cfg.get("classifier", {}).get("fail_preceding_k", 0))
    transitions, labels = _flatten_chains_and_labels(chains, pre_k=pre_k, reward_cfg=reward_cfg)

    states, actions = stack_state_action(transitions)
    if states.shape[0] == 0:
        raise RuntimeError("No valid transitions loaded for classifier training.")

    use_action_in_obs = bool(cfg.get("classifier", {}).get("concat_action_to_obs", True))
    if use_action_in_obs:
        x = np.concatenate([states, actions], axis=1).astype(np.float32)
    else:
        x = states.astype(np.float32)

    if labels.shape[0] != x.shape[0]:
        raise RuntimeError(
            f"Dataset length mismatch: labels={labels.shape[0]} vs samples={x.shape[0]}. "
            "Please check transition filtering in data loader."
        )

    input_dim = int(x.shape[1])
    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    model = FailureClassifier(input_dim=input_dim, hidden_dim=int(cfg["classifier"].get("hidden_dim", 256))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["classifier"].get("learning_rate", 5e-4)))

    pos_weight_val = float(cfg["classifier"].get("pos_weight", 2.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device))

    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    y_t = torch.tensor(labels, dtype=torch.float32, device=device)

    epochs = int(cfg["classifier"].get("epochs", 80))
    batch_size = int(cfg["classifier"].get("batch_size", 512))
    save_every = int(cfg["classifier"].get("save_every_epochs", 5))
    ckpt_every = int(cfg["classifier"].get("checkpoint_every_epochs", 10))
    smooth_window = int(cfg["classifier"].get("smooth_window", 10))
    val_split = float(cfg["classifier"].get("val_split", 0.1))
    test_split = float(cfg["classifier"].get("test_split", 0.1))

    train_idx, val_idx, test_idx = _stratified_split_indices(labels, val_split, test_split, int(cfg.get("seed", 0)))
    if train_idx.size == 0:
        raise RuntimeError("Empty train split for classifier. Reduce val_split/test_split.")

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    n = x.shape[0]
    idx = train_idx.copy()
    n_train = int(idx.shape[0])

    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        ep_losses = []
        ep_acc = []

        for st in range(0, n_train, batch_size):
            bidx = idx[st : st + batch_size]
            bx = x_t[bidx]
            by = y_t[bidx]

            logits = model(bx)
            loss = criterion(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                acc = (pred == by).float().mean().item()

            ep_losses.append(float(loss.item()))
            ep_acc.append(float(acc))

        train_losses.append(float(np.mean(ep_losses)))
        train_accs.append(float(np.mean(ep_acc)))

        v_loss, v_acc = _eval_on_subset(model, criterion, x_t, y_t, val_idx)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        if ep % save_every == 0:
            _save_metrics_csv(
                os.path.join(log_dir, "metrics_partial.csv"),
                (train_losses, val_losses),
                (train_accs, val_accs),
            )
            _save_curve(train_losses, "Classifier Train Loss", os.path.join(log_dir, "loss_partial.png"), smooth_window)
            _save_curve(val_accs, "Classifier Validation Accuracy", os.path.join(log_dir, "acc_partial.png"), smooth_window)

        if ep % ckpt_every == 0:
            torch.save(
                {
                    "epoch": ep,
                    "input_dim": input_dim,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(log_dir, f"checkpoint_epoch_{ep}.pt"),
            )

        if ep % 10 == 0:
            print(
                f"[classifier][epoch {ep}] "
                f"train_loss={train_losses[-1]:.4f} train_acc={train_accs[-1]:.4f} "
                f"val_loss={val_losses[-1]:.4f} val_acc={val_accs[-1]:.4f}"
            )

    test_loss, test_acc = _eval_on_subset(model, criterion, x_t, y_t, test_idx)
    train_cm = _confusion_on_subset(model, x_t, y_t, train_idx)
    val_cm = _confusion_on_subset(model, x_t, y_t, val_idx)
    test_cm = _confusion_on_subset(model, x_t, y_t, test_idx)

    torch.save(
        {"epoch": epochs, "input_dim": input_dim, "model_state_dict": model.state_dict()},
        os.path.join(log_dir, "model_final.pt"),
    )

    _save_metrics_csv(os.path.join(log_dir, "metrics.csv"), (train_losses, val_losses), (train_accs, val_accs))
    _save_confusion_csv(os.path.join(log_dir, "confusion_matrix.csv"), train_cm, val_cm, test_cm)
    _save_curve(train_losses, "Classifier Train Loss", os.path.join(log_dir, "loss.png"), smooth_window)
    _save_curve(val_accs, "Classifier Validation Accuracy", os.path.join(log_dir, "acc.png"), smooth_window)

    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": int(n),
                "num_train": int(train_idx.size),
                "num_val": int(val_idx.size),
                "num_test": int(test_idx.size),
                "positive_rate": float(np.mean(labels)),
                "input_dim": input_dim,
                "concat_action_to_obs": bool(use_action_in_obs),
                "fail_preceding_k": int(pre_k),
                "consecutive_fail_keep_k": int(fail_keep_k),
                "reward_cfg_path": str(reward_cfg_path),
                "val_split": float(val_split),
                "test_split": float(test_split),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "train_TP": int(train_cm["tp"]),
                "train_FP": int(train_cm["fp"]),
                "train_TN": int(train_cm["tn"]),
                "train_FN": int(train_cm["fn"]),
                "val_TP": int(val_cm["tp"]),
                "val_FP": int(val_cm["fp"]),
                "val_TN": int(val_cm["tn"]),
                "val_FN": int(val_cm["fn"]),
                "test_TP": int(test_cm["tp"]),
                "test_FP": int(test_cm["fp"]),
                "test_TN": int(test_cm["tn"]),
                "test_FN": int(test_cm["fn"]),
                "epochs": epochs,
                "device": str(device),
            },
            f,
            indent=2,
        )

    print(f"Saved classifier logs and model to: {log_dir}")


if __name__ == "__main__":
    main()

