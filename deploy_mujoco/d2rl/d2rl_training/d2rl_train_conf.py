# experiment_name = "2lane_400m_D2RL_Training_K2-fix-0.99_K1-max-1000-obs2"
# experiment_name = "2lane_400m_D2RL_Training_K2-min-0.975_K1-max-1000_reward-with-c"
# experiment_name = "2lane_400m_D2RL_Training_debug"

# experiment_name = "2lane_400m_D2RL_Training_K2-fix-0.99_K1-max-5000_reward-with-c-0.00005"
# experiment_name = "2lane_400m_D2RL_Training_K2-fix-0.99_K1-max-5000"

# root_folder = "D:/experiments_data/IIS/Dense-Deep-Reinforcement-Learning-main/data_analysis/raw_data/NADE_K1-10000/"
# root_folder = "/home1/rk19/Dense-Deep-Reinforcement-Learning-main/data_analysis/raw_data/NADE_K1-10000/"
root_folder = "/mnt/4TB/rk/Dense-Deep-Reinfocement-Learning-main/data_analysis/raw_data/NADE_K1-10000/"

data_folders = [
    "Experiment-2lane_400m_NADE-refuse_K1-10000_2024-06-13",
    "Experiment-2lane_400m_NADE-refuse_K1-10000_2024-06-19",
    "Experiment-2lane_400m_NADE-refuse_K1-10000_2024-07-10"
]
data_folder_weights = [1, 1, 1]

local_dir = "./ray_results/"
num_workers = 12
clip_reward_threshold = 100

scale_reward = 5000
P_min = 1e-7
P_max = 5e-5

K1_train = True
K1_fixed_value = 100
K1_action_space_range = [0, 1]
K1_max = 5000  # [0, 1] * K1_max + 1  ->  [1, K1_max + 1]

K2_train = False
K2_fixed_value = 0.99
K2_action_space_range = [0, 1]
K2_min = 0.80  # 1 - [0, 1] * (1 - K2_min)  ->  [1, K2_min]

c_reward_scale = 0.01
# c_reward_scale = 0

experiment_name = ("alpha-analysis-Training"
                   + "_K2-" + ("min" if K2_train else "fix") + "-" + str(K2_min if K2_train else K2_fixed_value)
                   + "_K1-" + ("max" if K1_train else "fix") + "-" + str(K1_max if K1_train else K1_fixed_value)
                   + "_reward-with-c-1-" + str(c_reward_scale))
                   # + (("_reward-with-c-1-" + str(c_reward_scale)) if c_reward_scale != 0 else ""))
