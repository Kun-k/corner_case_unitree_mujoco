# train_plum_blossom

Plum-blossom pile terrain generation and deployment test.

## Features
- Generate rectangular piles with configurable grid size, center, base height, and gap.
- Update selected pile heights online with `(ix, iy, delta_h)` controls.
- Lift robot root if terrain is raised abruptly to reduce penetration risk.
- Set robot spawn pose with `TerrainTrainer.set_robot_spawn_pose`.

## Run
```bash
python deploy/deploy_mujoco_go2/train_plum_blossom/test_plum_blossom.py
```

