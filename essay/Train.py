import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rl.Callbacks import TensorboardCallback
from rl.env_factory import make_env
from model.Vehicle import Vehicle
from model.SafeGuard import SafeGuardUtility
from model.Track import Track
from model.Task import Task
from utils.data_loader import (
    load_auxiliary_parking_areas,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_station_zp_positions,
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


# 学习率线性衰减
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


# 读取线路数据
slopes, slope_intervals = load_slopes()
speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True, dtype=np.float64)
aps, dps = load_auxiliary_parking_areas()
ly_zp, pa_zp = load_station_zp_positions()
levi_curves_list, brake_curves_list = load_safeguard_curves(
    "levi_curves_list", "brake_curves_list"
)

sgu = SafeGuardUtility(
    speed_limits=speed_limits,
    speed_limit_intervals=speed_limit_intervals,
    min_curves_list=levi_curves_list,
    max_curves_list=brake_curves_list,
    gamma=0.95,
)
track = Track(
    slopes=slopes,
    slope_intervals=slope_intervals,
    speed_limits=speed_limits,
    speed_limit_intervals=speed_limit_intervals,
    ASA_aps=aps,
    ASA_dps=dps,
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
model_save_path = "data/optimal/RL/ppo_mtto"
vecnormalize_save_path = "data/optimal/RL/vecnormalize.pkl"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 创建训练环境
venv_train = DummyVecEnv(
    [
        lambda: make_env(
            vehicle=vehicle,
            track=track,
            safeguardutility=sgu,
            task=task,
            gamma=reward_discount,
            max_step_distance=ds,
        )
    ]
)
venv_train = VecMonitor(venv_train)
venv_train = VecNormalize(
    venv=venv_train, norm_obs=False, norm_reward=True, gamma=reward_discount
)

model = PPO(
    "MlpPolicy",
    venv_train,
    device="cpu",
    verbose=1,
    learning_rate=linear_schedule(3e-4),
    n_steps=2048,
    batch_size=256,
    n_epochs=15,
    gamma=reward_discount,  # 提高折扣因子
    gae_lambda=0.95,
    clip_range=0.2,  # 缩小clip区间, 防止策略突变
    # clip_range_vf=None,
    # normalize_advantage=True,
    ent_coef=0.01,  # 增加探索
    vf_coef=0.5,  # 增加价值函数权重
    max_grad_norm=0.5,
    tensorboard_log="./mtto_ppo_tensorboard_logs/",  # TensorBoard日志存放目录
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # 改进网络结构
    ),
)


# 训练，并使用tensorboard记录回报和网络损失变化
model.learn(
    total_timesteps=200_000,
    callback=TensorboardCallback(),
    log_interval=5,
    tb_log_name="trainning_log",
)
model.save(model_save_path)
venv_train.save(vecnormalize_save_path)
venv_train.close()


print("Training finished.")
print(f"Model saved to: {model_save_path}.zip")
print(f"VecNormalize stats saved to: {vecnormalize_save_path}")
print("Run essay/Evaluate.py to evaluate the trained policy.")
