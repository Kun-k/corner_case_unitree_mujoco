import os.path

import numpy as np
import yaml

from deploy.deploy_mujoco_go2.terrain_trainer import TerrainTrainer
import argparse


def main():

    args = argparse.ArgumentParser()
    args.add_argument("--log_name", type=str, default="cmase_logs")
    argument = args.parse_args()

    current_path = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_path, "logs", argument.log_name)

    train_config_file = os.path.join(log_dir, "train_config.yaml")

    with open(train_config_file, "r", encoding="utf-8") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    go2_cfg = [train_config['go2_task'], train_config['go2_config']]
    trainer = TerrainTrainer(go2_cfg, f"train_CMA_ES/logs/{argument.log_name}/terrain_config.yaml")

    trainer.render = True
    trainer.start_viewer()

    best_angle_array_path = os.path.join(log_dir, "best_angle_array.npy")
    angle_array = np.load(best_angle_array_path)

    max_robot_steps = train_config['max_robot_steps']

    trainer.init_skip_time = 0
    trainer.init_skip_frame = 10
    trainer.reset()

    trainer.terrain_changer.generate_trig_terrain(angle_array)
    trainer.terrain_changer.enforce_safe_spawn_area(
        center_world=(0.0, 0.0),
        safe_radius_m=train_config['safe_radius_m'],
        blend_radius_m=train_config['blend_radius_m'],
        target_height=0.0,
    )

    trainer.viewer.update_hfield(trainer.terrain_changer.hfield_id)
    trainer.viewer.sync()

    total_reward = 0.0
    done = False

    for _ in range(max_robot_steps):
        _, _, reward, done, _ = trainer.step_only_robot()

        total_reward += float(reward)
        if done:
            break

    print(f"Total reward: {total_reward:.3f}, done={done}")
    trainer.close_viewer()


if __name__ == "__main__":
    main()
