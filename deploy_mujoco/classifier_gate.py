import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class FailureClassifier(nn.Module):
    """Legacy MLP classifier (matches train_offline/train_classifier.py)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FailureClassifierLN(nn.Module):
    """LayerNorm variant kept for backward compatibility with older checkpoints."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _normalize_state_dict(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def _infer_input_hidden_from_state_dict(state_dict: dict, default_hidden: int) -> Tuple[int, int]:
    # Expect first linear at net.0.weight with shape [hidden, input_dim].
    w = state_dict.get("net.0.weight", None)
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[1]), int(w.shape[0])
    return 0, int(default_hidden)


class ClassifierGate:
    """Loads a trained classifier and scores (obs, action) or obs-only inputs."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        hidden_dim: int = 1024,
        concat_action_to_obs: bool = True,
    ):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device)
        self.concat_action_to_obs = bool(concat_action_to_obs)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        raw_state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
        if raw_state_dict is None:
            raise ValueError("Classifier checkpoint missing model_state_dict/state_dict")

        state_dict = _normalize_state_dict(raw_state_dict)
        inferred_input_dim, inferred_hidden = _infer_input_hidden_from_state_dict(state_dict, default_hidden=int(hidden_dim))

        input_dim = int(ckpt.get("input_dim", inferred_input_dim))
        if input_dim <= 0:
            raise ValueError("Classifier checkpoint missing valid input_dim")

        # Use checkpoint-inferred hidden size to avoid config mismatch.
        hidden_dim = int(inferred_hidden if inferred_hidden > 0 else hidden_dim)

        load_errors = []
        model = FailureClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        try:
            model.load_state_dict(state_dict, strict=True)
            self.model = model
        except RuntimeError as e:
            load_errors.append(str(e))
            model_ln = FailureClassifierLN(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
            try:
                model_ln.load_state_dict(state_dict, strict=True)
                self.model = model_ln
            except RuntimeError as e2:
                load_errors.append(str(e2))
                raise RuntimeError(
                    "Failed to load classifier checkpoint with supported architectures.\n"
                    + "\n---\n".join(load_errors)
                )

        self.model.eval()
        self.input_dim = input_dim

    def _build_features(self, obs_batch: np.ndarray, act_batch: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_batch, dtype=np.float32)
        act = np.asarray(act_batch, dtype=np.float32)

        if obs.ndim == 1:
            obs = obs[None, :]
        if act.ndim == 1:
            act = act[None, :]

        if self.concat_action_to_obs:
            feat = np.concatenate([obs, act], axis=1).astype(np.float32)
        else:
            feat = obs.astype(np.float32)

        if feat.shape[1] != self.input_dim:
            raise ValueError(
                f"Classifier input dim mismatch: expected {self.input_dim}, got {feat.shape[1]}. "
                "Check concat_action_to_obs and checkpoint compatibility."
            )
        return feat

    def predict_proba(self, obs_batch: np.ndarray, act_batch: np.ndarray) -> np.ndarray:
        feat = self._build_features(obs_batch, act_batch)
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32, device=self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy().astype(np.float32)


def build_gate_from_cfg(cfg: dict, base_dir: str) -> Optional[ClassifierGate]:
    gate_cfg = cfg.get("classifier_gate", {}) if isinstance(cfg, dict) else {}
    if not bool(gate_cfg.get("enabled", False)):
        return None

    ckpt = str(gate_cfg.get("checkpoint_path", "")).strip()
    if not ckpt:
        raise ValueError("classifier_gate.enabled=true but classifier_gate.checkpoint_path is empty")
    if not os.path.isabs(ckpt):
        ckpt = os.path.normpath(os.path.join(base_dir, ckpt))

    return ClassifierGate(
        checkpoint_path=ckpt,
        device=str(gate_cfg.get("device", cfg.get("device", "cpu"))),
        hidden_dim=int(gate_cfg.get("hidden_dim", 1024)),
        concat_action_to_obs=bool(gate_cfg.get("concat_action_to_obs", True)),
    )

