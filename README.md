# MTTO

中高速磁浮列车运行速度曲线优化 —— 动态规划（基线）& 强化学习（主）双链路。

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [脚本详解](#脚本详解)
  - [RL 训练 · `train_rl`](#rl-训练--train_rl)
  - [RL 评估 · `evaluate_rl`](#rl-评估--evaluate_rl)
  - [训练日志分析 · `analyze_training_data`](#训练日志分析--analyze_training_data)
  - [DP 基线复现 · `reproduce_dp`](#dp-基线复现--reproduce_dp)
  - [DP 结果可视化 · `show_dp_result`](#dp-结果可视化--show_dp_result)
  - [RL 结果可视化 · `show_rl_result`](#rl-结果可视化--show_rl_result)
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
| RL 评估 | `python -m scripts.evaluate_rl` |
| 训练日志分析 | `python -m scripts.analyze_training_data` |
| DP 基线复现 | `python -m scripts.reproduce_dp` |
| DP 结果可视化 | `python -m scripts.show_dp_result` |
| RL 结果可视化 | `python -m scripts.show_rl_result` |
| 防护曲线可视化 | `python -m scripts.show_safeguard_curves` |
| 计算并保存防护曲线 | `python -m scripts.calc_and_save_safeguard_curves` |
| 最短运行时间曲线 | `python -m scripts.calc_min_operation_time_curve` |
| 实际运营数据展示 | `python -m scripts.show_real_operation_data` |
| 奖励函数可视化 | `python -m scripts.show_reward_function` |
| 势函数可视化 | `python -m scripts.show_potential_function` |

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
| `--vec-env-type` | `str` | `subproc` | 向量化后端：`dummy` / `subproc`（仅 `num-envs>1` 时 `subproc` 生效） |
| `--subproc-start-method` | `str` | `spawn` | 多进程启动方式：`spawn` / `forkserver` |
| `--max-step-distance` | `float` | `100.0` | 相邻状态转移间的最大移动距离 (m) |
| `--schedule-time-s` | `float` | `440.0` | 规划运行时间 (s) |

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
| `--tb-log-name` | `str` | `trainning_log` | TensorBoard 日志子目录名 |
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

#### Best-Eval（训练期最优轨迹评估）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-best-eval` | `bool` | 取决于 run-mode | 启用训练期最优轨迹评估 |
| `--best-eval-trigger-mode` | `str` | `steps` | 触发模式：`steps`（按训练步数）/ `episodes`（按完成回合数） |
| `--best-eval-trigger-interval` | `int` | `100000` | 评估触发间隔 |
| `--best-eval-deterministic` | `bool` | `True` | 是否使用确定性策略推理 |

评估规则：优先判断是否成功到达 → 比较总 reward → 停车误差 → 时刻误差 → 能耗。
每次刷新最优时，在 `output/optimal/rl/best_{trigger_mode}/` 下保存模型、轨迹和指标。

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

### RL 评估 · `evaluate_rl`

加载训练好的 PPO 模型，在单环境中执行评估 rollout，可选录制视频。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load-dir` | `str` | `output/optimal/rl/final/` | 模型与 VecNormalize 文件所在目录 |
| `--reward-discount` | `float` | `0.99` | 折扣因子（重建环境用） |
| `--step-distance` | `float` | `100.0` | 环境最大步距 (m) |
| `--device` | `str` | `cpu` | 推理设备 |
| `--deterministic` | `bool` | `True` | 是否使用确定性策略 |
| `--record-video` | `bool` | `False` | 是否录制评估视频 |
| `--save-trajectory` | `bool` | `True` | 是否保存轨迹 NPZ 与指标 JSON |
| `--video-folder` | `str` | `mtto_eval_video` | 视频输出目录 |
| `--output-dir` | `str` | `None` | 轨迹文件输出目录（默认回退到 `--load-dir`） |
| `--video-length` | `int` | `10000` | 最大录制步数 |
| `--video-trigger-step` | `int` | `0` | 视频录制触发步数 |
| `--enable-env-diagnostics` | `bool` | `False` | 启用环境诊断信息采集 |

```bash
# 默认评估
python -m scripts.evaluate_rl

# 录制视频
python -m scripts.evaluate_rl --record-video

# 指定模型目录与设备
python -m scripts.evaluate_rl \
    --load-dir output/optimal/rl/.../final/ \
    --device cuda
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

加载已保存的 RL 最优轨迹及指标，叠加防护曲线背景渲染。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--curve-dir` | `str` | `output/optimal/rl/best` | 递归搜索曲线文件的目录 |
| `--no-safeguard` | `flag` | — | 不绘制防护曲线背景 |
| `--factor` | `float` | `0.99` | 防护曲线渲染因子 |

```bash
python -m scripts.show_rl_result
python -m scripts.show_rl_result --curve-dir output/optimal/rl/best_steps
python -m scripts.show_rl_result --no-safeguard
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