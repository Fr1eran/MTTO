# MTTO

中高速磁浮列车运行速度曲线优化 —— 动态规划（基线）& 强化学习（主）双链路。

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [脚本详解](#脚本详解)
  - [RL 训练 · `train_rl`](#rl-训练--train_rl)
  - [RL 消融训练 · `train_reward_ablation`](#rl-消融训练--train_reward_ablation)
  - [RL 评估 · `evaluate_rl`](#rl-评估--evaluate_rl)
  - [训练日志分析 · `analyze_training_data`](#训练日志分析--analyze_training_data)
  - [DP 基线复现 · `reproduce_dp`](#dp-基线复现--reproduce_dp)
  - [DP 结果可视化 · `show_dp_result`](#dp-结果可视化--show_dp_result)
  - [RL 结果可视化 · `show_rl_result`](#rl-结果可视化--show_rl_result)
  - [DP 与 RL 对比可视化 · `compare_rl_dp`](#dp-与-rl-对比可视化--compare_rl_dp)
    - [SPS 合规分析 · `analyze_sps_compliance`](#sps-合规分析--analyze_sps_compliance)
  - [RL 消融结果可视化 · `show_reward_ablation`](#rl-消融结果可视化--show_reward_ablation)
  - [防护曲线 · `show_safeguard_curves`](#防护曲线--show_safeguard_curves)
  - [计算并保存防护曲线 · `calc_and_save_safeguard_curves`](#计算并保存防护曲线--calc_and_save_safeguard_curves)
  - [最短运行时间曲线 · `calc_min_operation_time_curve`](#最短运行时间曲线--calc_min_operation_time_curve)
  - [实际运营数据 · `show_real_operation_data`](#实际运营数据--show_real_operation_data)
  - [奖励函数展示 · `show_reward_function`](#奖励函数展示--show_reward_function)
  - [势函数展示 · `show_potential_function`](#势函数展示--show_potential_function)
- [测试](#测试)

---

## 项目结构

```
MTTO/
├── model/                  # 核心模型
│   ├── common/             #   能耗计算 (ECC)、最短运行时间参考 (ORS)
│   ├── force/              #   制动力、运行阻力
│   ├── ocs/                #   防护曲线、安全工具、停车点步进、运营任务
│   ├── track/              #   线路信息
│   └── vehicle/            #   车辆参数
├── rl/                     # 强化学习
│   ├── callbacks.py        #   训练回调（TensorBoard 日志 & 最优轨迹评估）
│   ├── env_factory.py      #   环境工厂
│   ├── evaluation.py       #   评估辅助
│   ├── experiment_utils.py #   reward profile、运行元数据、输出命名
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
| RL 消融训练 | `python -m scripts.train_reward_ablation` |
| RL 评估 | `python -m scripts.evaluate_rl` |
| 训练日志分析 | `python -m scripts.analyze_training_data` |
| DP 基线复现 | `python -m scripts.reproduce_dp` |
| DP 结果可视化 | `python -m scripts.show_dp_result` |
| RL 结果可视化 | `python -m scripts.show_rl_result` |
| DP 与 RL 对比可视化 | `python -m scripts.compare_rl_dp` |
| SPS 合规分析 | `python -m scripts.analyze_sps_compliance` |
| RL 消融结果可视化 | `python -m scripts.show_reward_ablation` |
| 防护曲线可视化 | `python -m scripts.show_safeguard_curves` |
| 计算并保存防护曲线 | `python -m scripts.calc_and_save_safeguard_curves` |
| 最短运行时间曲线 | `python -m scripts.calc_min_operation_time_curve` |
| 实际运营数据展示 | `python -m scripts.show_real_operation_data` |
| 奖励函数可视化 | `python -m scripts.show_reward_function` |
| 势函数可视化 | `python -m scripts.show_potential_function` |

RL 工作流脚本 `train_rl`、`train_reward_ablation`、`evaluate_rl`、`analyze_training_data`、`show_rl_result`、`show_reward_ablation` 统一支持 `--dry-run`，用于预览有效配置、路径解析结果、运行矩阵或展示计划，而不执行训练/评估/分析/绘图。

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
| `--num-envs` | `int` | `1` | 并行采样环境数量 |
| `--vec-env-type` | `str` | `subproc` | 向量化后端：`dummy` / `subproc`（仅 `num-envs>1` 时 `subproc` 生效 |
| `--max-step-distance` | `float` | `100.0` | 相邻状态转移间的最大移动距离 (m) |
| `--schedule-time-s` | `float` | `430.0` | 规划运行时间 (s) |


#### 奖励配置与实验标识

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--reward-profile` | `str` | `full_shaping` | 奖励预设：`basic`、`basic_safety`、`basic_safety_stopping`、`basic_safety_stopping_punctuality`、`full_shaping` |
| `--experiment-tag` | `str` | `None` | 附加实验标签，用于隔离输出目录与 TensorBoard 运行名 |

`basic` 固定包含 `energy + comfort`；其余预设仅沿 `safety / stopping / punctuality` 三个 shaping 维度逐级打开。

#### PPO 超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--reward-discount` | `float` | `0.99` | 回报折扣因子 γ |
| `--rollout-steps-per-update` | `int` | `2048` | 每次更新的 rollout 总步数 |
| `--n-steps-per-env` | `int` | 自动推导 | 每个环境的步数（优先级高于 `--rollout-steps-per-update`） |
| `--total-timesteps` | `int` | `200000` | 训练总步数 |
| `--device` | `str` | `cpu` | 运行设备：`cpu` / `cuda` |

#### 日志与分析

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tensorboard-log-dir` | `str` | `mtto_ppo_tensorboard_logs` | TensorBoard 日志根目录 |
| `--tb-log-name` | `str` | 自动生成 | TensorBoard 运行名称；未指定时会拼接 run-mode、reward-profile、时间参数和 experiment-tag |
| `--log-interval` | `int` | `1`（tune）/ `5`（reproduce）/ `1`（monitor_best）/ `10`（best_only） | PPO 日志打印间隔 |
| `--tb-sample-interval-steps` | `int` | `1` | 回调最小采样步长 |
| `--env-diagnostics-interval-steps` | `int` | 同 `--tb-sample-interval-steps` | 环境诊断快照记录间隔 |
| `--force-dump-interval-steps` | `int` | `0`（关闭） | 按步长强制刷新 TensorBoard 缓冲区 |
| `--tb-batch-dump-records` | `int` | `0`（关闭） | 按采样记录数批量刷新 TensorBoard 缓冲区 |
| `--output-root` | `str` | `output/optimal/rl/` | 训练结果输出根目录 |
| `--run-mode` | `str` | `tune` | `tune` / `reproduce` / `monitor_best` / `best_only` |
| `--enable-tb` | `bool` | 取决于 run-mode | 启用 TensorBoard 日志 |
| `--enable-callback` | `bool` | 取决于 run-mode | 启用 TensorBoard 回调 |
| `--enable-monitor` | `bool` | 取决于 run-mode | 启用 VecMonitor 包装器 |
| `--enable-env-diagnostics` | `bool` | 取决于 run-mode | 启用环境诊断信息采集 |
| `--enable-auto-analysis` | `bool` | 取决于 run-mode | 启用训练后自动分析 |
| `--dry-run` | `bool` | `False` | 仅解析有效训练配置、输出路径和运行元数据预览，不创建环境或启动训练 |

每次训练会在实验输出根目录写入 `run_metadata.json`，记录 reward profile、run-mode、TensorBoard 运行名、schedule_time_s、max_step_distance 等信息，供评估与轨迹展示脚本复用。

#### Best-Eval（训练期最优轨迹评估）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-best-eval` | `bool` | 取决于 run-mode | 启用训练期最优轨迹评估 |
| `--best-eval-trigger-mode` | `str` | `steps` | 触发模式：`steps`（按训练步数）/ `episodes`（按完成回合数） |
| `--best-eval-trigger-interval` | `int` | `100000` | 评估触发间隔 |
| `--best-eval-deterministic` | `bool` | `True` | 是否使用确定性策略推理 |

成功判定使用 `TrainService.max_stop_error` 与 `TrainService.max_arr_time_error_ratio`：
- `stop_error_m <= max_stop_error`
- `abs(time_error_s) / schedule_time <= max_arr_time_error_ratio`

Best-eval 排序规则：
- 一旦出现成功轨迹，所有成功轨迹都优先于所有未成功轨迹
- 在成功轨迹之间，优先比较总能耗，越小越优
- 如果当前还没有成功轨迹，才回退到按总 reward 比较
- 停站误差与绝对时间误差仅作为稳定 tie-break

每次刷新最优时，在实验目录下的 `best_{trigger_mode}/` 中保存模型、VecNormalize、`best_trajectory.npz` 与 `best_trajectory_metrics.json`。

如果后续要执行 PBRS 消融实验，建议统一使用 `monitor_best` 模式，以保留 rollout 基础监控和训练期最优轨迹评估，同时避免高频诊断采样带来的额外开销。

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

# 使用 basic+safety+stopping 预设，并附加实验标签
python -m scripts.train_rl --run-mode monitor_best --reward-profile basic_safety_stopping --experiment-tag exp_a

# 仅预览 monitor_best 训练配置与输出路径
python -m scripts.train_rl --run-mode monitor_best --reward-profile basic_safety --dry-run

# 低开销训练，仅保留 best-eval
python -m scripts.train_rl --run-mode best_only

# 430s tune + steps 触发 best-eval
python -m scripts.train_rl --output-root output/optimal/rl/ --schedule-time-s 430.0 --max-step-distance 100.0 --run-mode tune --total-timesteps 1000000 --num-envs 4 --vec-env-type subproc --tb-sample-interval-steps 10 --env-diagnostics-interval-steps 10 --tb-batch-dump-records 10240 --best-eval-trigger-mode steps --best-eval-trigger-interval 100000 --best-eval-deterministic --device cpu

# 430s tune + episodes 触发 best-eval
python -m scripts.train_rl --output-root output/optimal/rl/ --schedule-time-s 440.0 --max-step-distance 100.0 --run-mode tune --total-timesteps 1000000 --num-envs 4 --vec-env-type subproc --tb-sample-interval-steps 10 --env-diagnostics-interval-steps 10 --tb-batch-dump-records 10240 --best-eval-trigger-mode episodes --best-eval-trigger-interval 1000 --best-eval-deterministic --device cpu

# 430s monitor_best + safety_speed 输出
python -m scripts.train_rl --output-root output/optimal/rl/safety_speed/ --schedule-time-s 430.0 --max-step-distance 100.0 --run-mode monitor_best --total-timesteps 1000000 --num-envs 4 --vec-env-type subproc --best-eval-trigger-mode episodes --best-eval-trigger-interval 1000 --best-eval-deterministic --device cpu
```

---

### RL 消融训练 · `train_reward_ablation`

按四种 PBRS 奖励方案批量串行训练 PPO，并强制运行在 `monitor_best` 语义下。该脚本不是 `train_rl` 的全量参数代理，而是一个精简入口：只暴露 `monitor_best` 仍然有意义的训练参数；`--run-mode`、`--reward-profile`、`--output-root`、`--tb-log-name`、所有 `--enable-*` 开关、高频 callback flush 参数和自动分析参数都被隐藏并由脚本内部固定。

默认 reward case 顺序如下：
- `basic`
- `basic_safety`
- `basic_safety_stopping`
- `basic_safety_stopping_punctuality`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ablation-output-root` | `str` | `output/optimal/rl/ablation` | 消融批次输出根目录 |
| `--ablation-tag` | `str` | `None` | 批次标签；若存在重复训练，会自动附加 repeat 标识 |
| `--reward-profiles` | `list[str]` | 上述四种默认 case | 指定 reward case 子集或重排顺序 |
| `--repeats` | `int` | `1` | 每种情形的重复训练次数 |
| `--seed-list` | `list[int]` | `None` | 显式指定每次重复的 seed；指定后优先于 `--repeats` / `--base-seed` |
| `--base-seed` | `int` | `None` | 当 `repeats > 1` 时按 `base-seed + repeat_index` 推导各轮 seed |
| `--schedule-time-s` | `float` | `430.0` | 规划运行时间 |
| `--max-step-distance` | `float` | `100.0` | 相邻状态转移最大移动距离 |
| `--reward-discount` | `float` | `0.99` | 回报折扣因子 |
| `--num-envs` | `int` | `1` | 训练环境数量 |
| `--vec-env-type` | `str` | `subproc` | 向量化环境后端（启动方式自动选择） |
| `--rollout-steps-per-update` | `int` | `2048` | PPO rollout 步数 |
| `--n-steps-per-env` | `int` | 自动推导 | 每个环境的步数 |
| `--total-timesteps` | `int` | `200000` | 总训练步数 |
| `--tensorboard-log-dir` | `str` | `mtto_ppo_tb_logs` | TensorBoard 日志根目录 |
| `--log-interval` | `int` | 沿用 `monitor_best` 默认 | PPO 日志打印间隔 |
| `--best-eval-trigger-mode` | `str` | `steps` | 训练期 best-eval 触发模式 |
| `--best-eval-trigger-interval` | `int` | `100000` | 训练期 best-eval 触发间隔 |
| `--best-eval-deterministic` | `bool` | `True` | best-eval 是否使用确定性策略 |
| `--device` | `str` | `cpu` | 运行设备 |
| `--dry-run` | `bool` | `False` | 仅展开 reward case × repeat × seed 的运行矩阵，不启动训练 |

消融脚本沿用相同策略：当启用 `subproc` 且并行环境数大于 1 时，自动优先 `forkserver`，否则使用 `spawn`。

批次执行过程中会在 `--ablation-output-root` 下维护 `ablation_manifest.json`，记录每个 case / repeat 的输出路径、TensorBoard 运行名、run metadata 路径和执行状态，供后续 `show_reward_ablation` 直接消费。

```bash
# 默认四种 case，各跑一次
python -m scripts.train_reward_ablation --base-seed 42

# 两次重复训练，按同一基准 seed 派生 42 / 43
python -m scripts.train_reward_ablation \
    --ablation-output-root output/optimal/rl/ablation_exp_1 \
    --ablation-tag exp_1 \
    --repeats 2 \
    --base-seed 42

# 只跑两个 case，并预览运行矩阵
python -m scripts.train_reward_ablation \
    --reward-profiles basic basic_safety_stopping \
    --seed-list 7 8 \
    --dry-run
```

---

### RL 评估 · `evaluate_rl`

加载训练好的 PPO 模型，在单环境中执行评估 rollout，可选录制视频。若 `--load-dir` 所在实验目录存在 `run_metadata.json`，评估会优先复用其中的 `schedule_time_s`、`reward_discount`、`max_step_distance` 与 `reward_profile`；只有在显式传参时才覆盖这些值。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load-dir` | `str` | `output/optimal/rl/final/` | 模型与 VecNormalize 文件所在目录 |
| `--reward-discount` | `float` | 从 `run_metadata.json` 读取，否则 `0.99` | 折扣因子（重建环境用） |
| `--schedule-time-s` | `float` | 从 `run_metadata.json` 读取，否则 `430.0` | 规划运行时间 |
| `--step-distance` | `float` | 从 `run_metadata.json` 读取，否则 `100.0` | 环境最大步距 (m) |
| `--reward-profile` | `str` | 从 `run_metadata.json` 读取，否则 `full_shaping` | 评估所使用的奖励预设 |
| `--device` | `str` | `cpu` | 推理设备 |
| `--deterministic` | `bool` | `True` | 是否使用确定性策略 |
| `--record-video` | `bool` | `False` | 是否录制评估视频 |
| `--save-trajectory` | `bool` | `True` | 是否保存轨迹 NPZ 与指标 JSON |
| `--video-folder` | `str` | `mtto_eval_video` | 视频输出目录 |
| `--output-dir` | `str` | `None` | 轨迹文件输出目录（默认回退到 `--load-dir`） |
| `--video-length` | `int` | `10000` | 最大录制步数 |
| `--video-trigger-step` | `int` | `0` | 视频录制触发步数 |
| `--enable-env-diagnostics` | `bool` | `False` | 启用环境诊断信息采集 |
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

# 覆盖训练元数据中的 reward profile 与时间参数
python -m scripts.evaluate_rl \
    --load-dir output/optimal/rl/.../final/ \
    --reward-profile basic_safety_stopping \
    --schedule-time-s 430.0

# 仅预览有效评估配置
python -m scripts.evaluate_rl --load-dir output/optimal/rl/.../final/ --dry-run
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

加载已保存的 DP 最优速度曲线及指标，叠加防护曲线背景渲染。

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
- 训练期间按步数评估得到的 `best_steps`
- 训练期间按回合评估得到的 `best_episodes`
- 训练结束后单独评估得到的 `final`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--curve-dir` | `str` | `output/optimal/rl` | RL 输出根目录或任一实验子目录 |
| `--trajectory-source` | `str` | `best` | 轨迹来源：`best` / `best_steps` / `best_episodes` / `final` |
| `--no-safeguard` | `flag` | — | 不绘制防护曲线背景 |
| `--factor` | `float` | `0.99` | 防护曲线渲染因子 |
| `--dry-run` | `bool` | `False` | 仅解析将加载的轨迹产物与 metrics 路径，不读取数据或显示图窗 |

```bash
python -m scripts.show_rl_result
python -m scripts.show_rl_result --curve-dir output/optimal/rl --trajectory-source best_steps
python -m scripts.show_rl_result --curve-dir output/optimal/rl --trajectory-source best_episodes
python -m scripts.show_rl_result --curve-dir output/optimal/rl --trajectory-source final
python -m scripts.show_rl_result --curve-dir output/optimal/rl --trajectory-source final --dry-run
python -m scripts.show_rl_result --no-safeguard
```

---

### DP 与 RL 对比可视化 · `compare_rl_dp`

在同一窗口内对比展示 DP 与 RL 所选最优轨迹，包含三联图：
- 速度-位置最优轨迹叠加
- 加速度随位置变化曲线
- 累计总能耗（牵引+悬浮）随位置变化曲线

该脚本默认行为：
- DP 轨迹从 `output/optimal/dp` 递归搜索最新 `optimized_speed_curve.npz`
- RL 轨迹从 `output/optimal/rl` 递归搜索，默认 `trajectory_source=best`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dp-curve-dir` | `str` | `output/optimal/dp` | DP 输出根目录，递归搜索最新曲线产物 |
| `--rl-curve-dir` | `str` | `output/optimal/rl` | RL 输出根目录，递归搜索匹配轨迹产物 |
| `--trajectory-source` | `str` | `best` | RL 轨迹来源：`best` / `best_steps` / `best_episodes` / `final` |
| `--no-safeguard` | `flag` | — | 速度图不渲染 safeguard 背景 |
| `--factor` | `float` | `0.99` | safeguard 背景渲染因子 |

```bash
# 默认：自动选择最新 DP 与 RL(best) 轨迹并显示三联图
python -m scripts.compare_rl_dp

# 指定 RL 轨迹来源为最终评估轨迹
python -m scripts.compare_rl_dp --trajectory-source final

# 指定 DP/RL 搜索目录
python -m scripts.compare_rl_dp \
    --dp-curve-dir output/optimal/dp \
    --rl-curve-dir output/optimal/rl

# 关闭 safeguard 背景并调整渲染因子
python -m scripts.compare_rl_dp --no-safeguard --factor 0.97
```

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
| `--trajectory-source` | `str` | `best` | RL 轨迹来源：`best` / `best_steps` / `best_episodes` / `final` |
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

### RL 消融结果可视化 · `show_reward_ablation`

读取 `train_reward_ablation` 生成的 `ablation_manifest.json`，并完成两类展示：
- 平均回合奖励与平均回合长度曲线
- 最终或“最优”轨迹

曲线数据直接来自每个 run 的 `final/episode_metrics.npz`，即训练期间由 `EpisodeMetricsCollector` 收集的 `steps`、`ep_mean_reward`、`ep_mean_len`。轨迹展示默认按 reward profile 分子图绘制，且每个子图标题不承载详细指标，而是在左上角使用 SCI 风格的 panel label，例如 `(a)`、`(b)`、`(c)`、`(d)`。与轨迹有关的详细指标会统一输出到终端摘要，包括 panel label 与 reward profile 的映射、选中的 repeat/seed、success、total_reward、total_energy_j、time_error_s、stop_error_m 以及 artifact 路径。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ablation-root` | `str` | `output/optimal/rl/ablation` | 消融批次输出根目录，必须包含 `ablation_manifest.json` |
| `--trajectory-source` | `str` | `best` | 轨迹来源：`best` / `best_steps` / `best_episodes` / `final` |
| `--curve-layout` | `str` | `overlay` | 曲线布局：`overlay` / `separate` |
| `--trajectory-layout` | `str` | `separate` | 轨迹布局：首版固定按 reward profile 分子图展示 |
| `--reward-profiles` | `list[str]` | `None` | 仅展示指定 reward profile 子集 |
| `--no-safeguard` | `flag` | — | 轨迹图不绘制 safeguard 背景 |
| `--factor` | `float` | `0.99` | safeguard 背景渲染因子 |
| `--dry-run` | `bool` | `False` | 仅解析 manifest、episode_metrics 路径、候选轨迹与代表 repeat 选择结果，不加载数组或弹图窗 |

代表轨迹的 repeat 选择规则复用训练期“最优轨迹”的既有语义：
- 成功轨迹优先于未成功轨迹
- 成功轨迹之间按总能耗更低优先
- 若都未成功，则按总 reward 更高优先

```bash
# 默认展示：曲线叠加 + best 轨迹分子图
python -m scripts.show_reward_ablation --ablation-root output/optimal/rl/ablation

# 展示最终轨迹
python -m scripts.show_reward_ablation \
    --ablation-root output/optimal/rl/ablation \
    --trajectory-source final

# 只展示两个 reward profile，并把曲线切为 separate 布局
python -m scripts.show_reward_ablation \
    --ablation-root output/optimal/rl/ablation \
    --reward-profiles basic basic_safety \
    --curve-layout separate

# 仅预览将加载的曲线文件与代表轨迹，不弹图窗
python -m scripts.show_reward_ablation \
    --ablation-root output/optimal/rl/ablation \
    --dry-run
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

基于 ORS（Operation Reference System）计算从起点到终点的理论最短运行时间曲线。

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

### 奖励函数展示 · `show_reward_function`

可视化 RL 环境中的奖励函数曲线（能耗舒适度奖励、对标停车奖励、准点奖励）。

```bash
python -m scripts.show_reward_function
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
