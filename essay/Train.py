import json
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rl.Callbacks import TensorboardCallback
from rl.MTTOEnv import MTTOEnv
from model.Vehicle import Vehicle
from model.SafeGuard import SafeGuardUtility
from model.Track import Track
from model.Task import Task
from gymnasium.wrappers import FlattenObservation, RecordVideo, RecordEpisodeStatistics
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

# 创建训练环境
mttoenv_train = MTTOEnv(
    vehicle=vehicle,
    track=track,
    safeguardutil=sgu,
    task=task,
    gamma=reward_discount,
    max_step_distance=ds,
)

# 创建评估环境
mttoenv_eval = MTTOEnv(
    vehicle=vehicle,
    track=track,
    safeguardutil=sgu,
    task=task,
    gamma=reward_discount,
    max_step_distance=ds,
    render_mode="rgb_array",
)

# 记录训练过程的性能数据
num_eval_episodes_during_train = 1000

mttoenv_train = RecordEpisodeStatistics(
    FlattenObservation(mttoenv_train),
    buffer_length=num_eval_episodes_during_train,
)

# mttoenv_train = Monitor(
#     FlattenObservation(mttoenv_train),
# )


# 记录训练后的运行轨迹
mttoenv_eval = RecordVideo(
    FlattenObservation(mttoenv_eval),
    video_folder="mtto_eval_video",
    name_prefix="eval",
    episode_trigger=lambda x: True,
    fps=10,
)

model = PPO(
    "MlpPolicy",
    mttoenv_train,
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
    tensorboard_log="./mtto_ppo_tensorboard_logs/",  # TensorBoard日志存放目录
    # policy_kwargs=dict(
    #     net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),  # 改进网络结构
    # ),
)


# 训练，并使用tensorboard记录回报和网络损失变化
model.learn(
    total_timesteps=50_000,
    callback=TensorboardCallback(),
    log_interval=1,
    tb_log_name="first_log",
)

# 打印训练过程的性能数据
print("\nTrain Evaluation Summary:")

# 绘制训练过程的性能数据
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 总回报变化曲线
axes[0].plot(list(mttoenv_train.return_queue), linewidth=1.5)
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].set_title("Episode Reward Over Training")
axes[0].grid(True, alpha=0.3)

# 回合长度变化曲线
axes[1].plot(list(mttoenv_train.length_queue), linewidth=1.5, color="orange")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Episode Length")
axes[1].set_title("Episode Length Over Training")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


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
obs, info = mttoenv_eval.reset()
episode_over = False

while not episode_over:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = mttoenv_eval.step(action)
    reward_after += float(reward)
    episode_over = terminated or truncated

mttoenv_eval.close()  # 必须调用，否则无法保存视频

print("Evaluate after train:")
print(f"reward_after: {reward_after}")
