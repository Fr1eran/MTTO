# MTTO

中高速磁浮列车运行速度曲线优化 —— 动态规划（基线）& 强化学习（主）双链路。

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [脚本详解](#脚本详解)
  - [RL 训练 · `train_rl`](#rl-训练--train_rl)
  - [奖励消融实验 · `run_reward_ablation`](#奖励消融实验--run_reward_ablation)
  - [RL 评估 · `evaluate_rl`](#rl-评估--evaluate_rl)
  - [RL 中途计划时间突变实验 · `run_schedule_time_change`](#rl-中途计划时间突变实验--run_schedule_time_change)
  - [训练日志分析 · `analyze_training_data`](#训练日志分析--analyze_training_data)
  - [DP 基线复现 · `reproduce_dp`](#dp-基线复现--reproduce_dp)
  - [DP 结果可视化 · `show_dp_result`](#dp-结果可视化--show_dp_result)
  - [RL 结果可视化 · `show_rl_result`](#rl-结果可视化--show_rl_result)
  - [DP 与 RL 对比可视化 · `compare_rl_dp`](#dp-与-rl-对比可视化--compare_rl_dp)
    - [SPS 合规分析 · `analyze_sps_compliance`](#sps-合规分析--analyze_sps_compliance)
  - [防护曲线 · `show_safeguard_curves`](#防护曲线--show_safeguard_curves)
  - [计算并保存防护曲线 · `calc_and_save_safeguard_curves`](#计算并保存防护曲线--calc_and_save_safeguard_curves)
  - [最短运行时间曲线 · `calc_min_operation_time_curve`](#最短运行时间曲线--calc_min_operation_time_curve)
  - [实际运营数据 · `show_real_operation_data`](#实际运营数据--show_real_operation_data)
  - [势函数展示 · `show_potential_function`](#势函数展示--show_potential_function)
- [测试](#测试)

---

## 项目结构

```
MTTO/
├── model/                  # 核心模型
│   ├── common/             #   能耗计算 (ECC)、最短运行时间参考函数 (ORS)
│   ├── force/              #   制动力、运行阻力
│   ├── ocs/                #   防护曲线、安全工具、停车点步进、运营任务
│   ├── track/              #   线路信息
│   └── vehicle/            #   车辆参数
├── rl/                     # 强化学习
│   ├── callbacks.py        #   训练回调（TensorBoard 日志 & 最优轨迹评估）
│   ├── context_pool.py     #   DP 参考轨迹与不可变上下文池构建
│   ├── context_sampler.py  #   持有版本化分布的上下文采样器
│   ├── completion_critic.py #   任务完成度 Critic、缓冲区与课程回调
│   ├── dspdl.py            #   DSPDL 配置、统计器、分布求解器与回调
│   ├── env_factory.py      #   环境工厂
│   ├── evaluation.py       #   评估辅助
│   ├── experiment_utils.py #   reward preset、运行元数据、输出命名
│   ├── mtto_env.py         #   Gym 环境
│   └── training_analysis/  #   训练日志分析流水线
├── scripts/                # 可执行入口
├── tests/                  # 单元测试
├── data/                   # 线路 & 运营数据
├── output/                 # 输出产物（模型、曲线、报告）
└── utils/                  # 工具函数（绘图、IO、几何、索引）
```

---

## 快速开始

所有脚本均通过 `python -m scripts.<name>` 运行：

| 用途 | 命令 |
|------|------|
| RL 训练 | `python -m scripts.train_rl` |
| 奖励消融训练 | `python -m scripts.run_reward_ablation train` |
| RL 评估 | `python -m scripts.evaluate_rl` |
| RL 中途计划时间突变实验 | `python -m scripts.run_schedule_time_change evaluate` / `python -m scripts.run_schedule_time_change show` |
| 训练日志分析 | `python -m scripts.analyze_training_data` |
| DP 基线复现 | `python -m scripts.reproduce_dp` |
| DP 结果可视化 | `python -m scripts.show_dp_result` |
| RL 结果可视化 | `python -m scripts.show_rl_result` |
| 速度曲线对比可视化 | `python -m scripts.compare_speed_profiles` |
| SPS 合规分析 | `python -m scripts.analyze_sps_compliance` |
| 奖励消融结果展示 | `python -m scripts.run_reward_ablation show` |
| 防护曲线可视化 | `python -m scripts.show_safeguard_curves` |
| 计算并保存防护曲线 | `python -m scripts.calc_and_save_safeguard_curves` |
| 最短运行时间曲线 | `python -m scripts.calc_min_operation_time_curve` |
| 实际运营数据展示 | `python -m scripts.show_real_operation_data` |
| 势函数可视化 | `python -m scripts.show_potential_function` |

RL 工作流脚本 `train_rl`、`run_reward_ablation train/show`、`evaluate_rl`、`run_schedule_time_change evaluate`、`analyze_training_data` 和 `show_rl_result` 统一支持 `--dry-run`，用于预览有效配置、路径解析结果、运行矩阵或展示计划，而不执行训练、评估、分析或绘图。

---

## 脚本详解

### RL 训练 · `train_rl`

使用 PPO 算法训练磁浮列车最优速度曲线策略。通过 `--run-mode` 一键切换日志与分析开关。

#### 运行模式

| 模式 | 说明 |
|------|------|
| `tune`（默认） | 启用 TensorBoard、采样回调、best-eval、训练后自动分析 |
| `reproduce` | 关闭所有日志与分析，最大化训练效率 |
| `monitor_best` | 关闭高频采样回调，保留 VecMonitor 基础监控与 best-eval（rollout 指标写入 TensorBoard） |
| `best_only` | 仅保留 best-eval，适合低开销筛选最优模型 |

#### 训练环境与并行

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-envs` | `int` | `8` | 向量化采样环境数量 |
| `--vec-env-type` | `str` | `dummy` | 向量化后端：`dummy` / `subproc`；默认使用低开销的 `DummyVecEnv` |
| `--step-distance` | `float` | `30.0` | 固定空间控制步长 (m)，`--max-step-distance` 为兼容别名 |
| `--schedule-time-s` | `float` | `465.0` | 规划运行时间 (s) |

训练入口与奖励、方法、步长消融脚本共享上述环境默认值。显式指定
`--vec-env-type subproc` 时仍会启用多进程后端；消融输出目录中如果已经存在训练配置
不兼容的 manifest，新训练会拒绝覆盖，历史结果展示不受影响。

#### DSPDL 课程学习

通过 `--curriculum-profile dspdl --reference-curve-dir <dp-output-dir>` 启用离散型
SPDL 课程。代码层统一使用 DSPDL 命名：父进程读取 `ReferenceTrajectory`，由
`ContextPoolBuilder` 对 DP 轨迹采样并重建为只读 `ContextPool`，再将同一逻辑任务池
交给所有训练环境。每个训练环境只创建轻量的 `ContextSampler` 和
`DSPDLEpisodeAccumulator`，不会重复读取或重建 DP 轨迹。

每个环境在本地累计当前分布版本的上下文计数与折扣回报；`DSPDLCallback` 仅在课程
更新时统一拉取统计，使用缓存的完整任务 observation tensor 估值，再调用
`DSPDLDistributionSolver` 更新分布。采样器的 `sample()` 只负责按内部权重抽样，权重
及版本校验集中在更新接口中。达到目标分布阈值后，课程分布永久冻结，并释放各环境的
统计缓冲区。

使用 `--curriculum-profile dspdl_completion` 可启用基于任务完成度的 DSPDL。该配置
保留相同的上下文池、初始/目标分布、KL 信赖域和更新周期，但以独立的
`CompletionCritic` 预测 $[0,1]$ 区间内的任务完成度。每个完整回合的全部决策状态使用
同一个终局完成度进行监督训练；网络结构、优化器、学习率调度、batch size、epoch 数和
梯度裁剪均从 PPO value 分支解析，输出层改为 Sigmoid。迁移强度使用回合完成度 EMA，
不会写入 PPO 模型或单独保存 Completion Critic checkpoint。训练时可在 TensorBoard
查看 `completion/loss`、`completion/explained_variance`、
`completion/learning_rate` 和 `dspdl/alpha`。

配对实验表明，过大的迁移上限会使课程过早进入目标分布；当前
`CompletionDSPDLConfig.alpha_max` 默认取 `0.05`。可通过
`--completion-alpha-max <value>` 在独立实验中覆盖该值，覆盖结果会写入运行元数据。


#### 奖励配置与实验标识

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--reward-preset` | `str` | `basic_safety` | 奖励预设：`basic`、`basic_safety` |
| `--experiment-tag` | `str` | `None` | 附加实验标签，用于隔离输出目录与 TensorBoard 运行名 |

`basic` 固定包含 `energy + comfort`，`basic_safety` 在此基础上启用安全 PBRS。停站精度与准点要求只通过成功到站后的终端奖励和评估指标表达，不参与势函数塑形。

#### PPO 超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--reward-discount` | `float` | `0.995` | 回报折扣因子 γ |
| `--rollout-steps-per-update` | `int` | `2048` | 每次更新的 rollout 总步数 |
| `--n-steps-per-env` | `int` | 自动推导 | 每个环境的步数（优先级高于 `--rollout-steps-per-update`） |
| `--total-timesteps` | `int` | `200000` | 训练总步数 |
| `--device` | `str` | `cpu` | 运行设备：`cpu` / `cuda` |

#### 日志与分析

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tensorboard-log-dir` | `str` | `mtto_ppo_tensorboard_logs` | TensorBoard 日志根目录 |
| `--tb-log-name` | `str` | 自动生成 | TensorBoard 运行名称；未指定时会拼接 run-mode、reward-preset、时间参数和 experiment-tag |
| `--log-interval` | `int` | `1`（tune）/ `5`（reproduce）/ `1`（monitor_best）/ `10`（best_only） | PPO 日志打印间隔 |
| `--output-root` | `str` | `output/optimal/rl/` | 训练结果输出根目录 |
| `--run-mode` | `str` | `tune` | `tune` / `reproduce` / `monitor_best` / `best_only` |
| `--enable-tb` | `bool` | 取决于 run-mode | 启用 TensorBoard 日志 |
| `--enable-monitor` | `bool` | 取决于 run-mode | 启用 VecMonitor 包装器 |
| `--enable-auto-analysis` | `bool` | 取决于 run-mode | 启用训练后自动分析 |
| `--enable-safety-truncation-histogram` | `bool` | tune 模式启用 | 按 rollout 汇总安全截断位置并保存直方图 |
| `--dry-run` | `bool` | `False` | 仅解析有效训练配置、输出路径和运行元数据预览，不创建环境或启动训练 |

每次训练会写入 `run_metadata.json`，并在 `final/` 下生成 `reward_diagnostics.npz`。该二进制产物保存完整 episode 奖励分量累计值及 rollout 级 transition 充分统计量；奖励占比与相关性由训练后分析模块计算，不再写入 TensorBoard event。

#### Best-Eval（训练期最优轨迹评估）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-best-evaluation-artifacts` | `bool` | 取决于 run-mode | 启用训练期最优模型与轨迹产物 |
| `--evaluation-interval-rollouts` | `int` | `12` | 每完成指定数量的 PPO rollouts 执行一次评估 |
| `--evaluation-deterministic` | `bool` | `True` | 是否使用确定性策略推理 |
| `--evaluation-history-path` | `str` | `None` | 可选的完整评估历史 NPZ 输出路径 |

评估仅在 rollout 边界触发，不再在每个训练 step 中检查，也不再支持按完成 episode 数调度。评估历史使用 `rollout_indices` 与 `training_steps` 同时标记评估时点；最优轨迹 metrics 使用 `evaluation_rollout_index`。

成功判定使用 `TrainService.max_stop_error` 与 `TrainService.max_arr_time_error_ratio`：
- `stop_error_m <= max_stop_error`
- `abs(time_error_s) / schedule_time <= max_arr_time_error_ratio`

Best-eval 排序规则：
- 一旦出现成功轨迹，所有成功轨迹都优先于所有未成功轨迹
- 在成功轨迹之间，优先比较总能耗，越小越优
- 如果当前还没有成功轨迹，才回退到按总 reward 比较
- 停站误差与绝对时间误差仅作为稳定 tie-break

每次刷新最优时，在实验目录下的 `best_rollouts/` 中保存模型、`best_trajectory.npz` 与 `best_trajectory_metrics.json`。

如果后续要执行 PBRS 消融实验，建议统一使用 `monitor_best` 模式，以保留 rollout 基础监控和训练期最优轨迹评估，同时避免高频诊断采样带来的额外开销。

当前 PBRS 仅包含安全势函数：`basic` 不启用势函数，`basic_safety` 启用安全势函数。停站与准点均只在成功到站时通过终端奖励计分，不包含对应的势函数或稠密奖励。`scripts/show_potential_function.py` 中保留的停站势函数仅用于独立可视化与设计分析，不接入训练奖励链路。

#### 训练后自动分析

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--analysis-output-root` | `str` | `mtto_train_reports` | 分析报告输出目录 |
| `--analysis-min-points-per-10k-steps` | `float` | `5.0` | 每万步最低样本数 |
| `--analysis-min-unique-episodes` | `int` | `100` | 最低唯一回合数 |
| `--analysis-max-mean-step-gap` | `float` | `2048.0` | 最大平均训练步间隔 |
| `--analysis-sampling-quality-mode` | `str` | `warn_only` | 采样质量闸门：`warn_only` / `strict_fail` |

输出产物（轻量模式）：`report.md` + `analysis_snapshot.json`。

#### 示例

```bash
# 默认调优训练
python -m scripts.train_rl --run-mode tune

# 高效复现（关闭日志，不进行 best model 评估，仅得到最终训练模型）
python -m scripts.train_rl --run-mode reproduce

# 关闭高频回调，保留基础监控 + best-eval
python -m scripts.train_rl --run-mode monitor_best

# 使用安全 PBRS 预设，并附加实验标签
python -m scripts.train_rl --run-mode monitor_best --reward-preset basic_safety --experiment-tag exp_a

# 仅预览 monitor_best 训练配置与输出路径
python -m scripts.train_rl --run-mode monitor_best --reward-preset basic_safety --dry-run

# 低开销训练，仅保留 best-eval
python -m scripts.train_rl --run-mode best_only

# 430s tune，每 12 个 rollouts 触发一次 best-eval
python -m scripts.train_rl --output-root output/optimal/rl/ --schedule-time-s 430.0 --step-distance 100.0 --run-mode tune --total-timesteps 1000000 --num-envs 8 --vec-env-type dummy --evaluation-interval-rollouts 12 --evaluation-deterministic --device cpu

# 430s monitor_best，每 6 个 rollouts 评估一次
python -m scripts.train_rl --output-root output/optimal/rl/safety_speed/ --schedule-time-s 430.0 --step-distance 100.0 --run-mode monitor_best --total-timesteps 1000000 --num-envs 8 --vec-env-type dummy --evaluation-interval-rollouts 6 --evaluation-deterministic --device cpu
```

---

### 奖励消融实验 · `run_reward_ablation`

统一完成奖励消融的训练、断点恢复、固定起点评估和结果展示。实验固定关闭 DSPDL，避免课程分布与奖励设计混杂；两组配置分别为 `basic` 和 `basic_safety`，用于单独衡量安全 PBRS 的效果，每组默认使用固定种子 `11 / 131`。新版实验使用独立根目录 `output/optimal/rl/reward_ablation_safety`，不会复用旧四组消融的 manifest。

训练采用低开销 `reproduce` 配置，保留 VecMonitor 和回合指标采集，关闭 TensorBoard、高频环境诊断、best-eval 与自动分析。每隔 `--evaluation-interval-rollouts` 个 rollouts 执行一次确定性真实起点评估，并在训练结束后保存最终策略评估。

```bash
# 展开默认 2 × 2 运行矩阵，不写文件
python -m scripts.run_reward_ablation train --dry-run

# 训练全部奖励组；中断后执行同一命令会跳过产物完整的 completed 运行
python -m scripts.run_reward_ablation train \
    --output-root output/optimal/rl/reward_ablation_safety \
    --total-timesteps 1000000 \
    --num-envs 8

# 只补跑基础奖励与安全 PBRS
python -m scripts.run_reward_ablation train \
    --reward-presets basic basic_safety
```

批次根目录中的 `reward_ablation_manifest.json` 记录完整训练参数、每个 profile/seed 的产物路径和 `pending/running/completed/failed` 状态。重复运行仅跳过状态为 `completed` 且回合、周期评估、最终评估三类产物完整的任务；失败任务会被记录，但不会阻止其余组合继续运行。

`show` 子命令生成六面板学习图：回合奖励、固定起点成功率、停站误差、绝对时间误差、总能耗和舒适性，并可额外输出按 5 km 区间统计的安全违规箱线图。终端同时打印最终策略的均值、标准差和成功率。

```bash
python -m scripts.run_reward_ablation show \
    --output-file output/optimal/rl/reward_ablation_safety/learning_curves.png \
    --safety-output-file output/optimal/rl/reward_ablation_safety/safety_violations.png \
    --no-show

# 只检查 manifest 与可用产物
python -m scripts.run_reward_ablation show --dry-run
```

---

### RL 评估 · `evaluate_rl`

加载训练好的 PPO 模型，在单环境中执行评估 rollout，可选录制视频。若 `--load-dir` 所在实验目录存在 `run_metadata.json`，评估会优先复用其中的 `schedule_time_s`、`reward_discount`、`step_distance` 与 `reward_preset`；只有在显式传参时才覆盖这些值。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load-dir` | `str` | `output/optimal/rl/final/` | PPO 模型所在目录 |
| `--reward-discount` | `float` | 从 `run_metadata.json` 读取，否则 `0.995` | 折扣因子（重建环境用） |
| `--schedule-time-s` | `float` | 从 `run_metadata.json` 读取，否则 `430.0` | 规划运行时间 |
| `--step-distance` | `float` | 从 `run_metadata.json` 读取，否则 `30.0` | 环境固定空间控制步长 (m) |
| `--reward-preset` | `str` | 从 `run_metadata.json` 读取，否则 `basic_safety` | 评估所使用的奖励预设 |
| `--device` | `str` | `cpu` | 推理设备 |
| `--deterministic` | `bool` | `True` | 是否使用确定性策略 |
| `--record-video` | `bool` | `False` | 是否录制评估视频 |
| `--save-trajectory` | `bool` | `True` | 是否保存轨迹 NPZ 与指标 JSON |
| `--video-folder` | `str` | `mtto_eval_video` | 视频输出目录 |
| `--output-dir` | `str` | `None` | 轨迹文件输出目录（默认回退到 `--load-dir`） |
| `--video-length` | `int` | `10000` | 最大录制步数 |
| `--video-trigger-step` | `int` | `0` | 视频录制触发步数 |
| `--dry-run` | `bool` | `False` | 仅解析有效评估配置、训练元数据回填结果与输入输出路径，不加载模型或运行 rollout |

评估成功后可保存 `final_trajectory.npz` 与 `final_trajectory_metrics.json`。最终轨迹指标会显式写入 `trajectory_source=final`，并与训练期的最优轨迹共用同一套 selection metadata 结构。

```bash
# 默认评估
python -m scripts.evaluate_rl

# 录制视频
python -m scripts.evaluate_rl --record-video

# 指定模型目录与设备
python -m scripts.evaluate_rl \
    --load-dir output/optimal/rl/.../final/ \
    --device cuda

# 覆盖训练元数据中的 reward preset 与时间参数
python -m scripts.evaluate_rl \
    --load-dir output/optimal/rl/.../final/ \
    --reward-preset basic_safety \
    --schedule-time-s 430.0

# 仅预览有效评估配置
python -m scripts.evaluate_rl --load-dir output/optimal/rl/.../final/ --dry-run
```

---

### RL 中途计划时间突变实验 · `run_schedule_time_change`

批量评估同一 PPO 策略在“运行途中计划运行时间突然变化”场景下的响应，并展示多条速度曲线对比。脚本包含两个子命令：

- `evaluate`：加载模型，按多组时间变化量执行 rollout，并保存轨迹与指标。
- `show`：加载已保存的突变实验结果，绘制速度-距离对比图与安全防护背景。

默认批量运行 `Original`、`Minus 10s`、`Plus 10s`、`Minus 20s`、`Plus 20s` 五种情形。其中 `Original` 不触发计划时间变化，其余情形会在列车首次跨过 `--change-distance-m` 指定位置时调用环境的 `change_schedule_time()`。

#### `evaluate` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load-dir` | `str` | `output/optimal/rl/final/` | PPO 模型所在目录 |
| `--output-dir` | `str` | `output/optimal/rl/schedule_time_change_eval/` | 突变实验输出根目录；每次运行会创建时间戳子目录 |
| `--reward-discount` | `float` | 从 `run_metadata.json` 读取，否则 `0.995` | 折扣因子（重建环境用） |
| `--schedule-time-s` | `float` | 从 `run_metadata.json` 读取，否则 `430.0` | 突变前的初始规划运行时间 |
| `--step-distance` | `float` | 从 `run_metadata.json` 读取，否则 `30.0` | 环境固定空间控制步长 (m) |
| `--reward-preset` | `str` | 从 `run_metadata.json` 读取，否则 `basic_safety` | 评估所使用的奖励预设 |
| `--device` | `str` | `cpu` | 推理设备 |
| `--deterministic` | `bool` | `True` | 是否使用确定性策略 |
| `--change-distance-m` | `float` | `800.0` | 触发计划时间变化的位置 (m) |
| `--delta-times-s` | `str` | `0,-10,10,-20,20` | 逗号分隔的计划时间变化量；新计划时间为 `schedule_time_s + delta` |
| `--dry-run` | `bool` | `False` | 仅解析配置与路径，不加载模型或运行 rollout |

`evaluate` 模式会在输出目录中保存：

- `trajectory_{case}.npz`
- `trajectory_{case}_metrics.json`
- `schedule_time_change_summary.json`

#### `show` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load-dir` | `str` | `output/optimal/rl/schedule_time_change_eval/` | 突变实验结果目录；可指向根目录或单次时间戳实验目录 |
| `--save-figure` | `bool` | `True` | 是否保存对比图 |
| `--show` | `bool` | `True` | 是否弹出图窗 |
| `--figure-name` | `str` | `schedule_time_change_comparison.png` | 保存图像文件名 |
| `--factor` | `float` | `0.99` | 绘制安全防护背景时使用的安全系数 |

当 `--load-dir` 指向输出根目录时，`show` 会自动选择其中最新的时间戳实验目录；当它已经指向单次实验目录时，会直接加载该目录下的 `schedule_time_change_summary.json` 与轨迹文件。

```bash
# 仅预览将要运行的突变实验矩阵
python -m scripts.run_schedule_time_change evaluate --dry-run

# 使用训练得到的 final 模型运行默认五组突变实验
python -m scripts.run_schedule_time_change evaluate \
    --load-dir output/optimal/rl/.../final/ \
    --change-distance-m 800.0

# 自定义突变位置和时间变化组合
python -m scripts.run_schedule_time_change evaluate \
    --load-dir output/optimal/rl/.../final/ \
    --change-distance-m 12000.0 \
    --delta-times-s 0,-5,5,-15,15

# 展示最新一次突变实验结果，并保存对比图
python -m scripts.run_schedule_time_change show \
    --load-dir output/optimal/rl/schedule_time_change_eval/

# 只保存图像，不弹出图窗
python -m scripts.run_schedule_time_change show \
    --load-dir output/optimal/rl/schedule_time_change_eval/ \
    --no-show

# 展示某一次具体实验目录
python -m scripts.run_schedule_time_change show \
    --load-dir output/optimal/rl/schedule_time_change_eval/20260101_120000
```

---

### 训练日志分析 · `analyze_training_data`

对 TensorBoard 训练日志进行全维度分析并生成 LLM 友好报告。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--log-root` | `str` | `mtto_ppo_tensorboard_logs` | TensorBoard 日志根目录 |
| `--run-name` | `str` | 最新一次运行 | 指定运行子目录名 |
| `--output-root` | `str` | `mtto_train_reports` | 分析报告输出目录 |
| `--step-window-size` | `int` | `5000` | Step 快照窗口大小 |
| `--episode-window-size` | `int` | `20` | Episode 快照窗口大小 |
| `--ema-alpha` | `float` | `0.1` | 收敛分析 EMA 系数 |
| `--kl-threshold` | `float` | `0.03` | Approx KL 安全阈值 |
| `--near-miss-threshold-mps` | `float` | `1.0` | 安全边界近失阈值 (m/s) |
| `--position-bin-size-m` | `float` | `500.0` | 地理位置分箱大小 (m) |
| `--critical-point-radius-m` | `float` | `300.0` | SPS 区域邻域半径 (m) |
| `--top-k-spatial-bins` | `int` | `8` | 报告中空间风险 Top-K |
| `--top-k-critical-points` | `int` | `8` | 报告中关键点 Top-K |
| `--report-bar-width` | `int` | `24` | ASCII 柱状图宽度 |
| `--training-log-interval` | `int` | — | 训练日志间隔（存入元数据） |
| `--min-points-per-10k-steps` | `float` | `5.0` | 每万步最低样本数 |
| `--min-unique-episodes` | `int` | `100` | 最低唯一回合数 |
| `--max-mean-step-gap` | `float` | `2048.0` | 最大平均步间隔 |
| `--sampling-quality-mode` | `str` | `warn_only` | `warn_only` / `strict_fail` |
| `--export-csv` | `bool` | `False` | 导出 CSV 产物 |
| `--include-snapshots` | `bool` | `False` | 包含原始 step/episode 快照 |
| `--dry-run` | `bool` | `False` | 仅解析日志分析配置与输出路径，不执行分析 |

```bash
# 默认分析（轻量输出）
python -m scripts.analyze_training_data

# 指定运行 + 导出 CSV
python -m scripts.analyze_training_data \
    --run-name trainning_log_1 --export-csv

# 导出 CSV + 原始快照
python -m scripts.analyze_training_data \
    --export-csv --include-snapshots

# 严格采样质量闸门
python -m scripts.analyze_training_data \
    --sampling-quality-mode strict_fail

# 仅预览分析配置
python -m scripts.analyze_training_data --run-name trainning_log_1 --dry-run
```

---

### DP 基线复现 · `reproduce_dp`

基于动态规划（DP）+预计算状态转移图计算磁浮列车最优速度曲线。外层二分搜索调整时间乘子逼近目标运行时间，内层逆推 DP 求解最小能耗轨迹。

#### 优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output-root` | `str` | `output/optimal/dp` | 输出根目录 |
| `--schedule-time-s` | `float` | `430.0` | 规划运行时间 (s) |
| `--delta-speed-mps` | `float` | `0.1` | 速度搜索步长 (m/s) |
| `--max-outer-iterations` | `int` | `100` | 外层二分搜索最大迭代次数 |

> 输出目录规则：`{output-root}/{time}_{speed}_{division}/`，例如 430.0 s + 0.1 m/s + 变间距 30 子阶段 → `430p0_0p1_var30/`。

#### 阶段划分

支持两种离散化方式，通过 `--stage-division` 切换：

| 方式 | 说明 | 关联参数 |
|------|------|----------|
| `variable`（默认） | 基于安全临界点（IDP）划分大区间，每区间等分为 N 个子阶段 | `--sub-stage-count`（默认 `30`） |
| `uniform` | 从起点到终点按固定距离等分 | `--uniform-step-size`（默认 `100.0` m） |

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--stage-division` | `str` | `variable` | `variable` / `uniform` |
| `--sub-stage-count` | `int` | `30` | 变间距时每个临界区间的子阶段数 |
| `--uniform-step-size` | `float` | `100.0` | 等间距时的阶段步长 (m) |

#### 并行预计算

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--precompute-mode` | `str` | `serial` | `serial` / `parallel` |
| `--precompute-workers` | `int` | `CPU - 1` | 并行进程数 |
| `--precompute-chunk-size` | `int` | 自动估计 | 每个任务块的阶段数 |
| `--mp-start-method` | `str` | Windows 默认 `spawn` | `spawn` / `fork` / `forkserver` |
| `--hide-precompute-progress` | `flag` | — | 关闭预计算进度条 |

#### 磁盘缓存

状态转移图预计算结果可持久化到磁盘，避免相同参数下重复计算。缓存默认开启，位于 `output/_dp_transition_graph_cache/`。

**文件夹命名规则：** `{div_token}_{delta_token}_{hash12}`

| 组成部分 | 说明 | 示例 |
|----------|------|------|
| `div_token` | 变间距 `var{子阶段数}`，等间距 `uni{步长}` | `var30`、`uni100p0` |
| `delta_token` | 速度步长格式化值 | `0p1` |
| `hash12` | 所有输入参数的 SHA256 前 12 位 | `a1b2c3d4e5f6` |

> 完整示例：`var30_0p1_a1b2c3d4e5f6/`

**缓存文件夹内容：**

| 文件 | 用途 |
|------|------|
| `graph_data.pkl.gz` | gzip 压缩的完整状态转移图（stages、speed_states、transitions 等） |
| `metadata.json` | 人类可读的缓存元数据与 SHA256 完整性校验签名 |

**缓存键** 涵盖所有影响转移图计算的输入：离散化网格、车辆参数（mass、max_acc、max_dec 等）、ECC 能耗参数、防护曲线与限速、轨道坡度。任一输入变化会自动生成新的缓存文件夹。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--skip-disk-cache` | `flag` | — | 跳过磁盘缓存，每次强制重新计算 |

> 过期策略：手动删除对应的缓存文件夹即可，下次运行会自动重新计算并写入。

#### 示例

```bash
# 默认（变间距、串行预计算、启用磁盘缓存）
python -m scripts.reproduce_dp

# 并行预计算
python -m scripts.reproduce_dp --precompute-mode parallel

# 显式 4 进程 + 分块
python -m scripts.reproduce_dp \
    --precompute-mode parallel --precompute-workers 4 \
    --precompute-chunk-size 15 --mp-start-method spawn

# 等间距划分，步长 50 m
python -m scripts.reproduce_dp --stage-division uniform --uniform-step-size 50.0

# 变间距，增大子阶段密度
python -m scripts.reproduce_dp --stage-division variable --sub-stage-count 50

# 跳过磁盘缓存，强制重算
python -m scripts.reproduce_dp --skip-disk-cache

# 自定义时间 + 速度步长
python -m scripts.reproduce_dp --schedule-time-s 500.0 --delta-speed-mps 0.05
```

---

### DP 结果可视化 · `show_dp_result`

加载已保存的 DP 最优速度曲线及指标，叠加防护曲线背景渲染，并展示 DP 轨迹的冗余运行时间变化曲线。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--curve-dir` | `str` | `output/optimal/dp` | 递归搜索曲线文件的目录 |
| `--no-safeguard` | `flag` | — | 不绘制防护曲线背景 |
| `--factor` | `float` | `0.99` | 防护曲线渲染因子 |

```bash
python -m scripts.show_dp_result
python -m scripts.show_dp_result --curve-dir output/optimal/dp --factor 0.95
python -m scripts.show_dp_result --no-safeguard
```

---

### RL 结果可视化 · `show_rl_result`

加载已保存的 RL 单条轨迹及指标，叠加防护曲线背景渲染。该入口统一支持：
- 训练期间按 rollout 周期评估得到的 `best_rollouts`
- 训练结束后单独评估得到的 `final`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--curve-dir` | `str` | `output/optimal/rl` | RL 输出根目录或任一实验子目录 |
| `--trajectory-source` | `str` | `best` | 轨迹来源：`best` / `best_rollouts` / `final` |
| `--no-safeguard` | `flag` | — | 不绘制防护曲线背景 |
| `--factor` | `float` | `0.99` | 防护曲线渲染因子 |
| `--dry-run` | `bool` | `False` | 仅解析将加载的轨迹产物与 metrics 路径，不读取数据或显示图窗 |

```bash
python -m scripts.show_rl_result
python -m scripts.show_rl_result --curve-dir output/optimal/rl --trajectory-source best_rollouts
python -m scripts.show_rl_result --curve-dir output/optimal/rl --trajectory-source final
python -m scripts.show_rl_result --curve-dir output/optimal/rl --trajectory-source final --dry-run
python -m scripts.show_rl_result --no-safeguard
```

---

### 三基线速度曲线对比 · `compare_speed_profiles`

在同一窗口内对比展示 DP、RL 与实际运行的速度曲线，包含三联图：
- 速度-位置轨迹叠加（含安全防护背景）
- 加速度-位置曲线
- 累计总能耗（牵引+悬浮）-位置曲线

终端会以统一评价口径输出时间误差、停站误差、总能耗和 `comfort_tav` 的三基线对比表。实际运行曲线默认读取 `output/real_operation/aligned_real_operation_curve.npz`；首次使用前请先运行 `python -m scripts.transform_real_operation_curve`。

该脚本默认行为：
- DP 轨迹从 `output/optimal/dp` 递归搜索最新 `optimized_speed_curve.npz`
- RL 轨迹从 `output/optimal/rl` 递归搜索，默认 `trajectory_source=best`
- 实际运行轨迹从 `output/real_operation/aligned_real_operation_curve.npz` 读取

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dp-curve-dir` | `str` | `output/optimal/dp` | DP 输出根目录，递归搜索最新曲线产物 |
| `--rl-curve-dir` | `str` | `output/optimal/rl` | RL 输出根目录，递归搜索匹配轨迹产物 |
| `--real-curve` | `str` | `output/real_operation/aligned_real_operation_curve.npz` | 重标定后的实际运行曲线 NPZ |
| `--trajectory-source` | `str` | `best` | RL 轨迹来源：`best` / `best_rollouts` / `final` |
| `--no-safeguard` | `flag` | — | 速度图不渲染 safeguard 背景 |
| `--factor` | `float` | `0.99` | safeguard 背景渲染因子 |

```bash
# 默认：自动选择最新 DP、RL(best) 与实际运行轨迹，显示三联图和终端对比表
python -m scripts.compare_speed_profiles

# 指定 RL 轨迹来源为最终评估轨迹
python -m scripts.compare_speed_profiles --trajectory-source final

# 指定 DP/RL 搜索目录与实际运行曲线
python -m scripts.compare_speed_profiles \
    --dp-curve-dir output/optimal/dp \
    --rl-curve-dir output/optimal/rl \
    --real-curve output/real_operation/aligned_real_operation_curve.npz

# 关闭 safeguard 背景并调整渲染因子
python -m scripts.compare_speed_profiles --no-safeguard --factor 0.97
```

`python -m scripts.compare_rl_dp` 暂保留为兼容命令，但会提示迁移至新脚本。

---

### SPS 合规分析 · `analyze_sps_compliance`

离线回放 DP 与 RL 轨迹在停车点步进机制（SPS）下的合规性，核心判据固定为：
- 是否触发过步进请求（`triggered`）
- 是否存在“因未满足 `T_s` 时延约束导致的 min/max 防护边界违规”（`delay_related_boundary_violation`）

默认输出模式为 `text+plot`（文本摘要 + 主图）。主图仅在速度-位置平面展示，并标注：
- `REQUEST_START` 位置
- `STEP_COMPLETE` 位置

当事件点密集时，可切换为仅保留 marker（不显示文本注释）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dp-curve-dir` | `str` | `output/optimal/dp` | DP 输出根目录，递归搜索最新曲线产物 |
| `--rl-curve-dir` | `str` | `output/optimal/rl` | RL 输出根目录，递归搜索匹配轨迹产物 |
| `--trajectory-source` | `str` | `best` | RL 轨迹来源：`best` / `best_rollouts` / `final` |
| `--schedule-time-s` | `float` | `None` | 可选覆盖场景构建使用的计划运行时间 |
| `--step-delay-s` | `float` | `2.0` | SPS 回放中的步进平均时延 `T_s` |
| `--boundary-eps` | `float` | `1e-6` | 边界违规判定数值容差 |
| `--output-mode` | `str` | `text+plot` | 输出模式：`text` / `plot` / `json` / `text+plot` |
| `--json-output-path` | `str` | `None` | 启用 json 输出时，可选写入路径 |
| `--event-annotation` | `str` | `auto` | 标注模式：`auto` / `text` / `marker-only` |
| `--max-text-annotations` | `int` | `12` | `auto` 模式下文本标注上限 |
| `--no-safeguard` | `flag` | — | 主图不绘制 safeguard 背景 |
| `--factor` | `float` | `0.99` | safeguard 渲染与回放边界使用的因子 |

```bash
# 默认：文本摘要 + 主图（含事件 marker）
python -m scripts.analyze_sps_compliance

# 仅输出文本
python -m scripts.analyze_sps_compliance --output-mode text

# 输出 JSON 并写入文件
python -m scripts.analyze_sps_compliance \
    --output-mode json \
    --json-output-path output/optimal/sps_compliance_report.json

# 主图启用 marker-only（不显示文本注释）
python -m scripts.analyze_sps_compliance --event-annotation marker-only

# 指定 RL 轨迹来源与 SPS 时延参数
python -m scripts.analyze_sps_compliance \
    --trajectory-source final \
    --step-delay-s 2.0
```

---

### 防护曲线 · `show_safeguard_curves`

可视化展示磁浮列车运行安全防护曲线，包括 Levi 曲线、制动曲线、最小/最大速度约束、区间限速、辅助停车区、车站、加速区等。

```bash
python -m scripts.show_safeguard_curves
```

---

### 计算并保存防护曲线 · `calc_and_save_safeguard_curves`

离线计算并序列化保存完整的安全防护曲线数据至 `output/safeguardcurves/`，供后续训练与评估加载。

```bash
python -m scripts.calc_and_save_safeguard_curves
```

---

### 最短运行时间曲线 · `calc_min_operation_time_curve`

基于最短运行时间参考系统（Operation Reference System）的模块级函数计算
从起点到终点的理论最短运行时间曲线。

```bash
python -m scripts.calc_min_operation_time_curve
```

---

### 实际运营数据 · `show_real_operation_data`

加载并绘制上海磁浮示范线（龙阳路 → 浦东国际机场）的实际运营速度/加速度随里程变化曲线。

```bash
python -m scripts.show_real_operation_data
```

---

### 势函数展示 · `show_potential_function`

可视化安全势函数（Safety Speed / Safety Position / Safety Speed Adaptive），用于 RL reward shaping。

```bash
python -m scripts.show_potential_function
```

---

## 测试

```bash
# 全量测试
pytest

# 指定目录
pytest tests/
```
