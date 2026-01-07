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
from gymnasium.wrappers import FlattenObservation, RecordVideo
from stable_baselines3 import PPO


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
# 读取安全防护曲线
with open("data/rail/safeguard/levi_curves_list.pkl", "rb") as f:
    levi_curves_list = pickle.load(f)
with open("data/rail/safeguard/brake_curves_list.pkl", "rb") as f:
    brake_curves_list = pickle.load(f)

sgu = SafeGuardUtility(
    speed_limits=s_limits,
    speed_limit_intervals=s_intervals,
    min_curves_list=levi_curves_list,
    max_curves_list=brake_curves_list,
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
    start_position=ly_zp,
    start_speed=0.0,
    target_position=pa_zp,
    schedule_time=440.0,
    max_acc_change=0.75,
    max_arr_time_error=120,
    max_stop_error=0.3,
)

reward_discount = 0.99
ds = 100.0

maglevttoenv_train = MTTOEnv(
    vehicle=vehicle,
    track=track,
    safeguardutil=sgu,
    task=task,
    gamma=reward_discount,
    max_step_distance=ds,
)

maglevttoenv_eval = MTTOEnv(
    vehicle=vehicle,
    track=track,
    safeguardutil=sgu,
    task=task,
    gamma=reward_discount,
    max_step_distance=ds,
    render_mode="rgb_array",
)

maglevttoenv_train = FlattenObservation(maglevttoenv_train)
maglevttoenv_eval = RecordVideo(
    FlattenObservation(maglevttoenv_eval),
    video_folder="mtto_eval_video",
    name_prefix="eval",
    episode_trigger=lambda x: True,
    fps=10,
)

model = PPO(
    "MlpPolicy",
    maglevttoenv_train,
    device="cpu",
    verbose=1,
    learning_rate=1e-3,
    # n_steps=2048,
    # batch_size=64,
    # n_epochs=10,
    # gamma=reward_discount,  # 提高折扣因子
    # gae_lambda=0.95,
    clip_range=0.3,
    # clip_range_vf=None,
    # normalize_advantage=True,
    ent_coef=0.001,  # 增加探索
    # vf_coef=1.0,  # 增加价值函数权重
    # max_grad_norm=0.5,
    # policy_kwargs=dict(
    #     net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),  # 改进网络结构
    # ),
)


# Train
model.learn(total_timesteps=500_000)

user_input = (
    input("Training finished. Do you want to continue to evaluation? (y/n): ")
    .strip()
    .lower()
)
if user_input != "y":
    print("Exiting without evaluation.")
    sys.exit(0)

# Evaluate
# mean_reward_after, std_reward_after = evaluate_policy(
#     model, maglevttoenv_eval, n_eval_episodes=1, deterministic=True, render=True
# )

reward_after = 0.0
obs, info = maglevttoenv_eval.reset()
episode_over = False

while not episode_over:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = maglevttoenv_eval.step(action)
    reward_after += float(reward)
    episode_over = terminated or truncated

maglevttoenv_eval.close()
print("Evaluate after train:")
print(f"reward_after: {reward_after}")
