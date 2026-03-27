import ray
from ray import tune
import glob
from ray.tune.registry import register_env
from d2rl_training.d2rl_training_env import D2RLTrainingEnv
from tqdm import tqdm
import json
import time
from d2rl_training import d2rl_train_conf as d2rl_train_conf
import pandas as pd

currtime = time.localtime(time.time())

full_experiment_name = d2rl_train_conf.experiment_name + \
    "_"+time.strftime('%Y-%m-%d', currtime)

def env_creator(env_config):
    return D2RLTrainingEnv(d2rl_train_conf)


register_env("my_env", env_creator)
ray.init(include_dashboard=False, ignore_reinit_error=True)

import os
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import AgentID, PolicyID

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks


class MyCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.episode_data = []

    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.hist_data["constant"] = []
        episode.hist_data["weight_reward"] = []
        episode.hist_data["exposure"] = []
        episode.hist_data["positive_weight_reward"] = []
        episode.hist_data["episode_num"] = []
        episode.hist_data["step_num"] = []

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        last_info = episode.last_info_for()

        episode_data = {
            "episode_id": episode.episode_id,
            "episode_reward": episode.total_reward,
            "episode_length": episode.length,
        }
        self.episode_data.append(episode_data)

        df = pd.DataFrame(self.episode_data)
        df.to_csv(f"{d2rl_train_conf.local_dir + full_experiment_name}/episode_data.csv", index=False)

        # for key in episode.hist_data:
        #     episode.hist_data[key].append(last_info[key])
        # print(last_info)


def calculate_crash_weight(root_folder, path_list):
    for path in path_list:
        crash_unnorm_weight_dict = {}
        data_folder = root_folder + path
        crash_path = os.path.join(data_folder, "crash")
        print(crash_path)
        if os.path.isdir(crash_path):
            crash_data_path_list = glob.glob(crash_path + "/*.json")
        else:
            crash_data_path_list = None
            print("Crash data path not found!")

        # crash_unnorm_weight_list = []
        for crash_data_path in tqdm(crash_data_path_list):
            with open(crash_data_path) as crash_data:
                crash_data_json = json.load(crash_data)
                if crash_data_json["unnorm_weight_episode"] < 0.5:  # TODO: < 0.1
                    crash_unnorm_weight_dict[crash_data_path] = [
                        crash_data_json["unnorm_weight_episode"]
                    ]
        json_str = json.dumps(crash_unnorm_weight_dict, indent=4)
        with open(data_folder + "/" + "crash_unnorm_weight_dict.json", "w") as json_file:
            json_file.write(json_str)


def save_conf_to_location(source_file, target_location):
    # 读取源文件内容
    with open(source_file, 'r') as file:
        content = file.read()

    # 将内容写入目标位置
    with open(target_location, 'w') as file:
        file.write(content)

save_conf_location = (d2rl_train_conf.local_dir + full_experiment_name + '/d2rl_train_conf' + "_" +
                      time.strftime('%Y-%m-%d_%H-%M-%S', currtime) + '.py')
os.makedirs(os.path.dirname(save_conf_location), exist_ok=True)
save_conf_to_location('d2rl_training/d2rl_train_conf.py', save_conf_location)

calculate_crash_weight(d2rl_train_conf.root_folder, d2rl_train_conf.data_folders)
print("Nodes in the Ray cluster:")
print(ray.nodes())
tune.run(
    "PPO",
    stop={"training_iteration": 200},
    config={
        "env": "my_env",
        "num_gpus": 0,
        "num_workers": d2rl_train_conf.num_workers,
        "num_envs_per_worker": 1,
        "gamma": 1.0,
        "rollout_fragment_length": 600,
        "vf_clip_param": d2rl_train_conf.clip_reward_threshold,
        "framework": "torch",
        "ignore_worker_failures": True,
        "callbacks": MyCallbacks,
    },
    checkpoint_freq=1,
    local_dir=d2rl_train_conf.local_dir,
    name=full_experiment_name,
)

'''
conda activate d2rl_train
E:
cd E:/NNDE/Dense-Deep-Reinforcement-Learning-main
python d2rl_train.py
nohup python d2rl_train.py >/dev/null 2>&1 &
'''
