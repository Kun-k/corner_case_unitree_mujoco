# train_SAC_offline

Offline-only SAC training based on `train_SAC`.

This trainer only uses transitions loaded from offline `.pkl` files and runs gradient updates on the replay buffer. It does not interact with environment rollouts during optimization.

## Files

- `train.py`: offline training entry.
- `train_config.yaml`: config for offline training.
- `terrain_config.yaml`: reward/obs/terrain config used by reward recompute and env shape setup.

## Run

```bash
python deploy/deploy_mujoco_go2/train_SAC_offline/train.py
```

## Outputs

Under `deploy/deploy_mujoco_go2/train_SAC_offline/train_logs/<log_name>/`:

- `offline_metrics.csv`: update-level losses and entropy coeff logs.
- `checkpoint_update_<N>.zip`: periodic checkpoints.
- `model.zip`: final model.
- copied `train_config.yaml` and `terrain_config.yaml`.

## Notes

- `reward_recompute: true` uses `info` + `terrain_config.yaml` reward settings.
- `reward_recompute: false` uses stored transition reward in PKL.
- `consecutive_fail_keep_k` keeps only first K frames in each consecutive fail+stuck run.

