# TODO 梅花桩相关功能测试仍然不符合要求，需要继续修改
# TODO setconst


import numpy as np
from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer


def run_plum_blossom_demo():
    go2_cfg = ["terrain", "go2.yaml"]
    terrain_cfg = "train_plum_blossom/terrain_config.yaml"
    trainer = TerrainTrainer(go2_cfg, terrain_cfg)

    # 1) Generate plum-blossom piles centered at world origin.
    trainer.terrain_changer.generate_plum_blossom_piles(
        num_x=10,
        num_y=10,
        center_world=(0.0, 0.0),
        base_height=1.0,
        pile_size_m=0.2,
        gap_m=0,
    )
    if trainer.render:
        trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
        trainer.viewer.sync()

    # 2) Place robot above pile field center.
    trainer.set_robot_spawn_pose(x=0.0, y=0.0, z=0.45, yaw=0.0)

    # 3) Run and periodically adjust selected piles.
    rng = np.random.default_rng(0)
    total_steps = 120000
    for step in range(total_steps):
        if step > 0 and step % 150 == 0:
            controls = []
            for _ in range(5):
                ix = int(rng.integers(0, 6))
                iy = int(rng.integers(0, 6))
                delta_h = float(rng.uniform(-0.2, 0.2))
                controls.append((ix, iy, delta_h))
            trainer.terrain_changer.update_plum_blossom_piles(controls)
            if trainer.render:
                trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
                trainer.viewer.sync()

        # _, _, reward, done, info = trainer.step_only_robot()
        # if done:
        #     print(f"Episode terminated at step {step}. info={info}")
        #     break

    trainer.close_viewer()


if __name__ == "__main__":
    run_plum_blossom_demo()

