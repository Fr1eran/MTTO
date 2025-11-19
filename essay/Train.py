import json
import os
import sys
import pickle

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rl.MTTOEnv import MTTOEnv
from model.Vehicle import Vehicle
from model.SafeGuard import SafeGuardUtility
from model.Track import Track
from model.Task import Task
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# 读取线路数据
with open("data/rail/raw/slopes.json", "r", encoding="utf-8") as f:
    slope_data = json.load(f)
    slopes = slope_data["slopes"]
    slope_intervals = slope_data["intervals"]
with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
    sl_data = json.load(f)
    s_limits = sl_data["speed_limits"]
    s_limits = np.asarray(s_limits, dtype=np.float64) / 3.6
    s_intervals = sl_data["intervals"]
# 读取车站数据
with open("data/rail/raw/stations.json", "r", encoding="utf-8") as f:
    stations_data = json.load(f)
    ly_begin = stations_data["LY"]["begin"]
    ly_zp = stations_data["LY"]["zp"]
    ly_end = stations_data["LY"]["end"]
    pa_begin = stations_data["PA"]["begin"]
    pa_zp = stations_data["PA"]["zp"]
    pa_end = stations_data["PA"]["end"]
# 读取危险速度域数据
idp_points = np.load(file="data/rail/safeguard/idp_points.npy")
with open("data/rail/safeguard/levi_curves_part.pkl", "rb") as f:
    levi_curves_part = pickle.load(f)
with open("data/rail/safeguard/brake_curves_part.pkl", "rb") as f:
    brake_curves_part = pickle.load(f)

guard = SafeGuardUtility(
    speed_limits=s_limits,
    speed_limit_intervals=s_intervals,
    idp_points=idp_points,
    levi_curves_part_list=levi_curves_part,
    brake_curves_part_list=brake_curves_part,
    gamma=0.95,
)
track = Track(
    slopes=slopes,
    slope_intervals=slope_intervals,
    speed_limits=s_limits,
    speed_limit_intervals=s_intervals,
)
vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)
task = Task(
    starting_position=ly_zp,
    starting_velocity=0.0,
    destination=pa_zp,
    schedule_time=440.0,
    max_acc_change=0.75,
    max_arr_time_error=120,
    max_stop_error=0.3,
)

ds = 100.0

maglevttoenv_train = MTTOEnv(
    vehicle=vehicle,
    track=track,
    safeguardutil=guard,
    task=task,
    ds=ds,
)

maglevttoenv_eval = MTTOEnv(
    vehicle=vehicle,
    track=track,
    safeguardutil=guard,
    task=task,
    ds=ds,
    render_mode="human",
)

maglevttoenv_train = FlattenObservation(maglevttoenv_train)
maglevttoenv_eval = Monitor(FlattenObservation(maglevttoenv_eval))


model = PPO(
    "MlpPolicy",
    maglevttoenv_train,
    device="cpu",
    verbose=1,
    learning_rate=1e-3,  # 提高学习率
    n_steps=2048,
    batch_size=64,  # 增加批次大小
    n_epochs=10,
    gamma=0.995,  # 提高折扣因子
    gae_lambda=0.95,
    clip_range=0.3,
    clip_range_vf=None,
    normalize_advantage=True,
    ent_coef=0.08,  # 增加探索
    vf_coef=1.0,  # 增加价值函数权重
    max_grad_norm=0.5,
    # policy_kwargs=dict(
    #     net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),  # 改进网络结构
    # ),
)

# Evaluate before train
# mean_reward_before, std_reward_before = evaluate_policy(
#     model, maglevttoenv_eval, n_eval_episodes=10, render=False
# )

# print(r"Evaluate before train:")
# print(
#     r"mean_reward_before: {}, std_reward_before: {}",
#     mean_reward_before,
#     std_reward_before,
# )

# Train
model.learn(total_timesteps=50_000)

user_input = (
    input("Training finished. Do you want to continue to evaluation? (y/n): ")
    .strip()
    .lower()
)
if user_input != "y":
    print("Exiting without evaluation.")
    sys.exit(0)

# Evaluate
mean_reward_after, std_reward_after = evaluate_policy(
    model, maglevttoenv_eval, n_eval_episodes=1, deterministic=True, render=True
)

print("Evaluate after train:")
print(f"mean_reward_after: {mean_reward_after}, std_reward_after: {std_reward_after}")
