from gym import spaces, core
import os, glob
import random
import json
import numpy as np
import logging
from tqdm import tqdm
from d2rl_training import d2rl_train_conf as d2rl_train_conf
import copy


all_rewards = []

class D2RLTrainingEnv(core.Env):
    def __init__(self, d2rl_train_conf=d2rl_train_conf, is_debug=False):
        self.d2rl_train_conf = d2rl_train_conf
        self.is_debug = is_debug
        data_folders = [self.d2rl_train_conf.root_folder + folder for folder in self.d2rl_train_conf.data_folders]
        data_folder_weights = self.d2rl_train_conf.data_folder_weights
        if self.d2rl_train_conf.K1_train and self.d2rl_train_conf.K2_train:
            self.action_space = spaces.Box(low=np.array([self.d2rl_train_conf.K1_action_space_range[0], self.d2rl_train_conf.K2_action_space_range[0]], dtype=np.float32),
                                           high=np.array([self.d2rl_train_conf.K1_action_space_range[1], self.d2rl_train_conf.K2_action_space_range[1]], dtype=np.float32), shape=(2,))
        elif self.d2rl_train_conf.K1_train:
            self.action_space = spaces.Box(low=self.d2rl_train_conf.K1_action_space_range[0], high=self.d2rl_train_conf.K1_action_space_range[1], shape=(1,))
        elif self.d2rl_train_conf.K2_train:
            self.action_space = spaces.Box(low=self.d2rl_train_conf.K2_action_space_range[0], high=self.d2rl_train_conf.K2_action_space_range[1], shape=(1,))
        else:
            exit(0)

        # self.observation_space = spaces.Box(low=-5, high=5, shape=(10,))
        self.observation_space = spaces.Box(low=-5, high=5, shape=(7,))

        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0  # some customized metric logging
        self.total_episode, self.total_steps = 0, 0
        if isinstance(data_folders, list):
            data_folder = random.choices(data_folders, weights=data_folder_weights)[0]
        else:
            data_folder = data_folders
        self.crash_data_path_list, self.safe_data_path_list, self.crash_data_weight_list, self.crash_target_weight_list = self.get_path_list(
            data_folder)
        self.all_data_path_list = self.crash_data_path_list + self.safe_data_path_list
        self.episode_data_path = ""
        self.episode_data = None

        self.unwrapped.trials = 100
        self.unwrapped.reward_threshold = 1.5

        self.rewards = {}

    def get_path_list(self, data_folder):
        crash_target_weight_list = None
        if os.path.exists(data_folder + "/crash_unnorm_weight_dict.json"):
            with open(data_folder + "/crash_unnorm_weight_dict.json") as data_file:
                crash_unnorm_weight_dict = json.load(data_file)
                self.crash_unnorm_weight_dict = crash_unnorm_weight_dict
                crash_data_path_list = list(crash_unnorm_weight_dict.keys())
                crash_data_weight_list = [crash_unnorm_weight_dict[path][0] for path in crash_data_path_list]
        else:
            raise ValueError("No weight information!")
        tested_but_safe_path = os.path.join(data_folder, "tested_and_safe")
        if os.path.exists(data_folder + "/safe_weight_dict.json"):
            with open(data_folder + "/safe_weight_dict.json") as data_file:
                safe_weight_dict = json.load(data_file)
                safe_data_path_list = list(safe_weight_dict.keys())
        elif os.path.isdir(tested_but_safe_path):
            safe_data_path_list = glob.glob(tested_but_safe_path + "/*.json")
        else:
            safe_data_path_list = []
        logging.info(f'{len(crash_data_path_list)} Crash Events, {len(safe_data_path_list)} Safe Events')
        return crash_data_path_list, safe_data_path_list, crash_data_weight_list, crash_target_weight_list

    def reset(self, episode_data_path=None):
        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0
        self.total_episode = 0
        self.total_steps = 0
        self.episode_data_path = ""
        self.episode_data = None
        return self._reset(episode_data_path)

    def filter_episode_data(self, episode_data):
        invalid_timestep_list = []
        for timestep in episode_data["weight_step_info"]:
            if episode_data["weight_step_info"][timestep] < 1.001 and episode_data["weight_step_info"][timestep] > 0.9:
                invalid_timestep_list.append(timestep)
                logging.debug(f"popping out {episode_data['weight_step_info']}")
        for invalid_time_step in invalid_timestep_list:
            episode_data["weight_step_info"].pop(invalid_time_step, None)
            episode_data["criticality_step_info"].pop(invalid_time_step, None)
            episode_data["ndd_step_info"].pop(invalid_time_step, None)
            episode_data["drl_obs_step_info"].pop(invalid_time_step, None)
        # logging.debug(str(episode_data))
        return episode_data

    def sample_data_this_episode(self):
        if self.crash_data_weight_list:
            episode_data_path = random.choices(self.crash_data_path_list, weights=self.crash_data_weight_list)[0]
        else:
            raise ValueError("No weight information!")
        return episode_data_path

    def _reset(self, episode_data_path=None):
        self.total_episode += 1
        if not episode_data_path:
            self.episode_data_path = self.sample_data_this_episode()
        else:
            self.episode_data_path = episode_data_path
        with open(self.episode_data_path) as data_file:
            self.episode_data = self.filter_episode_data(json.load(data_file))
        if self.episode_data is not None:
            K_step_info = self.episode_data['K_step_info']
            self.episode_data['K_step_info_behavior'] = copy.deepcopy(K_step_info)
            self.episode_data['K_step_info_target'] = copy.deepcopy(K_step_info)
            all_obs = self.episode_data["drl_obs_step_info"]
            time_step_list = list(all_obs.keys())
            if len(time_step_list):
                init_obs = np.float32(all_obs[time_step_list[0]])[np.array([0, 1, 2, 4, 6, 7, 8])]
                return init_obs
            else:
                return self._reset()
        else:
            return self._reset()

    def step(self, action):
        obs = self._get_observation()
        done, _ = self._get_done()
        time_step_list = list(self.episode_data["drl_obs_step_info"].keys())
        if self.d2rl_train_conf.K1_train and self.d2rl_train_conf.K2_train:
            K_target = [action[0].item() * self.d2rl_train_conf.K1_max + 1, 1 - action[1].item() * (1 - self.d2rl_train_conf.K2_min)]
        elif self.d2rl_train_conf.K1_train:
            K_target = [action.item() * self.d2rl_train_conf.K1_max + 1, self.d2rl_train_conf.K2_fixed_value]
        else:
            K_target = [self.d2rl_train_conf.K1_fixed_value, 1 - action.item() * (1 - self.d2rl_train_conf.K2_min)]
        self.episode_data['K_step_info_target'][time_step_list[self.total_steps]] = K_target
        reward = self._get_reward()
        info = self._get_info()
        self.total_steps += 1
        return obs, reward, done, info

    def _get_info(self):
        return {}

    def close(self):
        return

    def _get_observation(self):
        all_obs = self.episode_data["drl_obs_step_info"]
        time_step_list = list(all_obs.keys())
        try:
            # obs = np.float32(all_obs[time_step_list[self.total_steps]])
            obs = np.float32(all_obs[time_step_list[self.total_steps]])[np.array([0, 1, 2, 4, 6, 7, 8])]
        except:
            print(self.total_steps, time_step_list)
            # obs = np.float32(all_obs[time_step_list[-1]])
            obs = np.float32(all_obs[time_step_list[-1]])[np.array([0, 1, 2, 4, 6, 7, 8])]
        return obs

    def get_multiple_adv_action_num(self, weight_info):
        adv_action_num = 0
        for timestep in weight_info:
            if weight_info[timestep] < 0.99:
                adv_action_num += 1
        return adv_action_num

    def _get_reward(self):  # ! Aim to remove the magnitude of the environment
        stop, reason = self._get_done()
        if not stop:
            return 0
        else:
            drl_weight = self._get_drl_weight(self.episode_data["unnorm_weight_step_info"],
                                              self.episode_data['K_step_info_behavior'],
                                              self.episode_data['K_step_info_target'])
            if 1 in reason:
                print('Behavior Action:', self.episode_data['K_step_info_behavior'])
                print('Target Action:', self.episode_data['K_step_info_target'])
                adv_action_num = self.get_multiple_adv_action_num(
                    self.episode_data["weight_step_info"])  # TODO: 暂时不改，应该没有影响
                if adv_action_num > 1:
                    return 0  # if multiple adversarial action is detected, this episode will be of no use
                clip_reward_threshold = self.d2rl_train_conf.clip_reward_threshold
                print("origin_reward:", drl_weight)
                q_amplifier_reward = clip_reward_threshold - drl_weight * self.d2rl_train_conf.scale_reward * clip_reward_threshold  # drl weight reward
                if q_amplifier_reward < -clip_reward_threshold:
                    q_amplifier_reward = -clip_reward_threshold
                elif q_amplifier_reward > clip_reward_threshold:
                    q_amplifier_reward = clip_reward_threshold
                all_rewards.append(q_amplifier_reward)
                print("final_reward:", q_amplifier_reward)
                self.rewards["final_reward"] = q_amplifier_reward
                return q_amplifier_reward
            else:
                return 0

    def _get_drl_weight(self, weight_info, K_step_info_behavior, K_step_info_target):
        total_q_amplifier_max, total_q_amplifier_min = 1, 1
        c_max, c_min = 1, 1
        for timestep in K_step_info_behavior:
            if timestep in weight_info:  # TODO: 去数据文件检查一下这个if什么时候成立
                K1_behavior, K2_behavior = K_step_info_behavior[timestep]
                if self.is_debug:
                    K1_target, K2_target = K1_behavior, K2_behavior
                else:
                    K1_target, K2_target = K_step_info_target[timestep]

                if weight_info[timestep] > 1:
                    unnorm_weight_behavior = 1 / K2_behavior
                    unnorm_weight_target = 1 / K2_target
                elif weight_info[timestep] < 0.999:
                    unnorm_weight_behavior = 1 / K1_behavior
                    unnorm_weight_target = 1 / K1_target
                else:
                    continue

                c_min_behavior = (K1_behavior - 1) * self.d2rl_train_conf.P_min + K2_behavior
                c_max_behavior = (K1_behavior - 1) * self.d2rl_train_conf.P_max + K2_behavior
                c_min_target = (K1_target - 1) * self.d2rl_train_conf.P_min + K2_target
                c_max_target = (K1_target - 1) * self.d2rl_train_conf.P_max + K2_target

                total_q_amplifier_min = total_q_amplifier_min * \
                                        unnorm_weight_behavior * c_max_behavior * \
                                        unnorm_weight_target * c_max_target
                total_q_amplifier_max = total_q_amplifier_max * \
                                        unnorm_weight_behavior * c_min_behavior * \
                                        unnorm_weight_target * c_min_target
                c_max *= c_max_target
                c_min *= c_min_target

        # clip_reward_threshold = self.d2rl_train_conf.clip_reward_threshold
        print('Rewards: ',
              total_q_amplifier_max * self.d2rl_train_conf.scale_reward +
              total_q_amplifier_min * self.d2rl_train_conf.scale_reward,
              abs(c_max - 1) * self.d2rl_train_conf.c_reward_scale * self.d2rl_train_conf.scale_reward +
              abs(c_min - 1) * self.d2rl_train_conf.c_reward_scale * self.d2rl_train_conf.scale_reward)
        self.rewards["reward_var_min"] = total_q_amplifier_min * self.d2rl_train_conf.scale_reward
        self.rewards["reward_var_max"] = total_q_amplifier_max * self.d2rl_train_conf.scale_reward
        self.rewards["reward_c_min"] = abs(c_min - 1) * self.d2rl_train_conf.c_reward_scale * self.d2rl_train_conf.scale_reward
        self.rewards["reward_c_max"] = abs(c_max - 1) * self.d2rl_train_conf.c_reward_scale * self.d2rl_train_conf.scale_reward
        return total_q_amplifier_max + total_q_amplifier_min + abs(c_max - 1) * self.d2rl_train_conf.c_reward_scale + abs(c_min - 1) * self.d2rl_train_conf.c_reward_scale

    def _get_done(self):
        stop = False
        reason = None
        if self.total_steps == len(self.episode_data["drl_obs_step_info"].keys()) - 1:
            stop = True
            if self.episode_data["collision_result"]:
                reason = {1: "CAV and BV collision"}
            else:
                reason = {4: "CAV safely exist"}
        return stop, reason

    def get_custom_metrics(self):
        return self.rewards


def calculate_crash_weight(root_folder, path_list):
    # self.d2rl_train_conf.P_min, self.d2rl_train_conf.P_max = P_range
    # K1, K2 = K

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
                # crash_unnorm_weight_list.append(crash_data_json["unnorm_weight_episode"])
                if crash_data_json["unnorm_weight_episode"] < 0.5:  # TODO: < 0.1
                    # timesteps = len(crash_data_json['unnorm_weight_step_info'])
                    # c_min = (K1 - 1) * self.d2rl_train_conf.P_min + K2
                    # c_max = (K1 - 1) * self.d2rl_train_conf.P_max + K2
                    crash_unnorm_weight_dict[crash_data_path] = [
                        crash_data_json["unnorm_weight_episode"],
                        # crash_data_json["unnorm_weight_episode"] * (c_max ** timesteps),
                        # crash_data_json["unnorm_weight_episode"] * (c_min ** timesteps),
                        # (self.d2rl_train_conf.P_min, self.d2rl_train_conf.P_max), (K1, K2)  # TODO: 写在这里没用，应该把K写进json，把P写成固定的（K暂时写固定）
                    ]
        json_str = json.dumps(crash_unnorm_weight_dict, indent=4)
        with open(data_folder + "/" + "crash_unnorm_weight_dict.json", "w") as json_file:
            json_file.write(json_str)


if __name__ == "__main__":
    calculate_crash_weight(d2rl_train_conf.root_folder, d2rl_train_conf.data_folders)

    env = D2RLTrainingEnv(is_debug=True)

    for i in range(100):
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break

    print(np.mean(all_rewards), np.min(all_rewards), np.max(all_rewards))

    import matplotlib.pyplot as plt
    import numpy as np

    # 生成一个示例一维数组
    data = np.array(all_rewards)  # 1000个随机数
    # data = data[data < 10000]

    # 绘制直方图
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')

    # 添加标题和标签
    plt.title('Histogram of Random Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # plt.xlim(0, 0.2)

    # 显示图形
    plt.show()
