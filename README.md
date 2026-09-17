# MTTO

中高速磁浮列车运行速度曲线优化 —— 动态规划（基线）& 强化学习（主）双链路。

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [脚本详解](#脚本详解)
  - [RL 训练 · `train_rl`](#rl-训练--train_rl)
  - [RL 评估 · `evaluate_rl`](#rl-评估--evaluate_rl)
  - [RL 中途计划时间突变实验 · `run_schedule_time_change`](#rl-中途计划时间突变实验--run_schedule_time_change)
  - [训练日志分析 · `analyze_training_data`](#训练日志分析--analyze_training_data)
  - [DP 基线复现 · `reproduce_dp`](#dp-基线复现--reproduce_dp)
  - [DP 结果可视化 · `show_dp_result`](#dp-结果可视化--show_dp_result)
  - [RL 结果可视化 · `show_rl_result`](#rl-结果可视化--show_rl_result)
  - [三基线速度曲线对比 · `compare_speed_profiles`](#三基线速度曲线对比--compare_speed_profiles)
    - [SPS 合规分析 · `analyze_sps_compliance`](#sps-合规分析--analyze_sps_compliance)
  - [线路环境与防护曲线 · `show_env_data`](#线路环境与防护曲线--show_env_data)
  - [计算并保存防护曲线 · `calc_and_save_safeguard_curves`](#计算并保存防护曲线--calc_and_save_safeguard_curves)
  - [最短运行时间曲线 · `calc_min_operation_time_curve`](#最短运行时间曲线--calc_min_operation_time_curve)
  - [实际运营数据 · `show_real_operation_data`](#实际运营数据--show_real_operation_data)
  - [势函数展示 · `show_potential_function`](#势函数展示--show_potential_function)
  - [终端评分函数可视化 · `show_score_function`](#终端评分函数可视化--show_score_function)
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
│   ├── env_factory.py      #   环境工厂
│   ├── evaluation.py       #   评估辅助
│   ├── experiment_utils.py #   reward preset、运行元数据、输出命名
│   ├── mtto_env.py         #   Gym 环境
│   └── training_analysis/  #   训练日志分析流水线
├── contracts/              # DTO、领域快照和版本化持久化 schema
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
| 方法消融实验 | `python -m scripts.run_method_ablation train/show` |
| 空间步长消融 | `python -m scripts.run_step_distance_ablation train/show` |
| RL 评估 | `python -m scripts.evaluate_rl` |
| RL 中途计划时间突变实验 | `python -m scripts.run_schedule_time_change evaluate` / `python -m scripts.run_schedule_time_change show` |
| 训练日志分析 | `python -m scripts.analyze_training_data` |
| DP 基线复现 | `python -m scripts.reproduce_dp` |
| DP 结果可视化 | `python -m scripts.show_dp_result` |
| RL 结果可视化 | `python -m scripts.show_rl_result` |
| 速度曲线对比可视化 | `python -m scripts.compare_speed_profiles` |
| SPS 合规分析 | `python -m scripts.analyze_sps_compliance` |
| 线路环境与防护曲线可视化 | `python -m scripts.show_env_data` |
| 计算并保存防护曲线 | `python -m scripts.calc_and_save_safeguard_curves` |
| 最短运行时间曲线 | `python -m scripts.calc_min_operation_time_curve` |
| 实际运营数据展示 | `python -m scripts.show_real_operation_data` |
| 势函数可视化 | `python -m scripts.show_potential_function` |
| 终端评分函数可视化 | `python -m scripts.show_score_function` |

RL 工作流脚本 `train_rl`、`evaluate_rl`、`run_schedule_time_change evaluate`、`analyze_training_data` 和 `show_rl_result` 统一支持 `--dry-run`，用于预览有效配置、路径解析结果或展示计划，而不执行训练、评估、分析或绘图。

论文全套仿真的固定参数、执行顺序、中断恢复和产物验收见
[完整仿真实验指导](output/完整仿真实验指导.md)。论文产物按
`00_figures`（仅存放方法与环境说明图）、`01_step_distance`、`02_method_ablation`、
`03_multiobjective`、`04_schedule_time_change`（分别存放对应实验的结果图与表格）分类，每批使用统一的
`YYYYMMDD_NN` 子目录。步长 v12、方法 v11、训练元数据 schema v6、消融 manifest schema v2、评估历史 schema v4 和严格评估
schema v2 不与旧结果混用。直接训练入口在目标目录已存在训练产物时会抛出 `FileExistsError` 拒绝直接覆盖。

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
| `--step-distance` | `float` | `30.0` | 固定空间控制步长 (m)，`--max-step-distance` 为兼容别名 |
| `--schedule-time-s` | `float` | `465.0` | 规划运行时间 (s) |

训练入口与奖励、方法及步长消融脚本统一使用 `DummyVecEnv`。
`--num-envs` 大于 1 时会在同一进程内依次采样多个环境，不再提供多进程后端选项。
消融输出目录中如果已经存在 manifest，新训练默认拒绝覆盖；配置不兼容或旧 schema 的
manifest 也不会用于恢复。需要断点恢复时显式使用 `--resume`；确认要重新开始时使用
`--force-new`，旧 manifest 会先备份。resume 只跳过状态为 `completed` 且 canonical 产物
完整的运行，失败、中断或产物不完整的运行会从头重跑。

#### 真实起点重置与防覆盖保护

所有强化学习训练环境始终在 reset 时重置到实际任务起点（`self.stepper.reset()`），完全移除了历史版本中的课程学习（DSPL）、状态池与 context 采样机制。训练入口不再接收 `--curriculum-profile` 或 `--reference-curve-dir` 参数。

为防止意外破坏已有的实验记录与模型权重，单次直接训练（`train_single_experiment`）在启动前会严格检查目标输出目录。如果目标目录中已经包含任何训练产物（如 `run_metadata_path`、`final_model_save_path`、`reward_diagnostics_path`、`metrics.json` 或 `trajectory.npz`），将直接抛出 `FileExistsError` 拒绝执行，不提供静默覆盖选项。如需重新训练，必须显式指定不同的 `--output-root` 或 `--experiment-tag`，或者在确认安全的前提下手动清理目标目录。

#### 奖励配置与实验标识

默认奖励预设为 `basic_safety_punctuality`，对应本文的物理先验奖励塑形
（Physics-Informed Reward Shaping, PIRS）：组合速度安全势与线性剩余裕度准点势。
若要进行方法消融或单独启用特定势函数，可通过 `--reward-preset` 指定：
- `basic`：仅包含基础 `energy + comfort`；
- `basic_safety`：在基础奖励上启用安全势函数塑形；
- `basic_punctuality`：在基础奖励上启用准点势函数塑形；
- `basic_safety_punctuality`：完整启用安全势与准点势（PIRS）。

令 `q` 为沿运行方向计算并裁剪到 `[0,1]` 的剩余距离比例，`b0` 为全程静止起点的
计划时间减最短运行时间，参考裕度为 `b_ref=b0*q`。
实际裕度与参考的差为 `e`，势函数为 `Phi=-K*e²/(sigma²+e²)`。安全势与准点势
在普通转移、成功终止、违规截断均统一计算为
`gamma*Phi(next)-Phi(previous)`，不将任务终止后的下一状态势显式归零。准点参数固定为
`K=5`、`sigma=20 s`，不提供运行时覆盖；负初始裕度保留符号，势函数不替代原终端
准点评分。
计划时间变化后以新计划更新全程参考线，不从变化点重锚；变更接口本身不发奖励，
不对跨外部计划变更的整段回报宣称固定任务策略不变性。

```bash
python -m scripts.train_rl --reward-preset basic_safety_punctuality
python -m scripts.show_potential_function --plot-type punctuality --no-show --output-dir output
```

诊断中记录 `punctuality_shaping` 分量，诊断 schema 为 4，读取旧版 2/3 时该分量补零。
上述参数是实验起点，短程训练验证不代表准点性能提升。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--reward-preset` | `str` | `basic_safety_punctuality` | 原始尺度奖励预设；支持 `basic`、`basic_safety`、`basic_punctuality`、`basic_safety_punctuality`（PIRS） |
| `--experiment-tag` | `str` | `None` | 附加实验标签，用于隔离输出目录与 TensorBoard 运行名 |

训练和评估均直接使用原始奖励尺度，停站精度和准点终端评分函数保持不变。

#### PPO 超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--reward-discount` | `float` | `0.998` | 回报折扣因子 γ |
| `--rollout-steps-per-update` | `int` | `8192` | 每次更新的 rollout 总步数 |
| `--n-steps-per-env` | `int` | 自动推导 | 每个环境的步数（优先级高于 `--rollout-steps-per-update`） |
| `--budget-mode` | `str` | `completed_episodes` | 训练预算模式：`completed_episodes` 或 `environment_steps` |
| `--training-episodes` | `int` | 模式默认 `5000` | 完成回合预算；仅用于 `completed_episodes` |
| `--training-rollouts` | `int` | `None` | PPO rollout 总数；`environment_steps` 模式必填 |
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

每次训练在 `final/` 写入 `policy.zip`、`metadata.json` 和 `episodes.npz`；最终独立评估再写入 `trajectory.npz` 与 `metrics.json`。`metadata.json` 的 `training_budget.mode` 记录预算模式，顶层不重复写入 `budget_mode`。该二进制产物保存完整 episode 奖励分量累计值及 rollout 级 transition 充分统计量。

#### Best-Eval（训练期最优轨迹评估）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-best-evaluation-artifacts` | `bool` | 取决于 run-mode | 启用训练期最优模型与轨迹产物 |
| `--evaluation-interval-rollouts` | `int` | `12` | 每完成指定数量的 PPO rollouts 执行一次评估 |
| `--evaluation-interval-episodes` | `int` | `None` | 每跨过指定数量的完成训练回合执行一次评估；与 rollout 间隔互斥 |
| `--evaluation-deterministic` | `bool` | `True` | 是否使用确定性策略推理 |
| `--evaluation-history-path` | `str` | `None` | 可选的完整评估历史 NPZ 输出路径 |

评估仅在 rollout 边界触发，不在每个训练 step 中执行。通用训练入口支持按 rollout
或按完成回合两种互斥调度；按回合调度在阈值被跨过后的下一次 `_on_rollout_start`
触发，因此实际回合数会有小幅 rollout 边界超调。评估历史同时记录规则目标回合数、
实际完成回合数、环境 transition 数与 rollout 序号；最优轨迹 metrics 使用
`evaluation_rollout_index`。

环境成功终止要求速度绝对值不超过 `0.01 m/s`，且
`stop_error_m <= max_stop_error * 30`（默认 `max_stop_error=0.3 m`，即 9 m）。
评估中的 `precise_arrival` 仍要求 `stop_error_m <= max_stop_error`，默认 0.3 m；
因此 success rate 与 precise-arrival rate 是不同指标。准点还要求
`abs(time_error_s) < max_arr_time_error_s`，默认阈值为 `10.0 s`。

安全判定统一为 `min_safety_margin_mps >= -1e-6` 且
`safety_violation_count == 0`；严格可行性要求成功、精确停站、准点和安全同时成立。
评估指标使用严格 schema v2，并显式保存 `safety_violation_count`、`safe` 与
`feasible`，不再从缺失字段的旧指标中重建选择键或安全状态。

Best-eval 排序规则：
- 严格可行（安全、成功、精确停站且准点）轨迹优先于所有不可行轨迹
- 严格可行轨迹之间优先选择总能耗更低者
- 尚无严格可行轨迹时，依次按安全成功、停站精度、准点性和能耗回退

论文方法消融固定为四组对照实验：PPO（`ppo`，对应 `basic`）、PPO+Safety（`ppo_safety`，对应 `basic_safety`）、PPO+Punctuality（`ppo_punctuality`，对应 `basic_punctuality`）与 PPO+PIRS（`ppo_pirs`，对应 `basic_safety_punctuality`）。
Manifest schema 版本为 2，protocol 版本为 11，不再要求 DP 参考轨迹。
步长消融（协议版本 12）统一使用完整 PIRS 基准（`ppo_pirs`），对 10、30、50、100 m 和 5 个种子统一使用 400 个 rollout（3,276,800 个环境状态转移），每 12 个 rollout 调度一次独立评估，周期评估点为 12–396（共 33 个点）。主图展示行程完成率和可行率（均值与样本标准差带，截断至 0–100%）。性能表汇总各步长 5 个种子 `best/` 轨迹的严格可行率（百分比与分子/分母）及各项指标均值 ± 样本标准差（保留全部 5 个种子），能耗单位为 kWh。旧步长协议结果不迁移或混入新输出目录。

固定 30 m 步长后的方法消融以 400 个 rollout（3,276,800 个环境状态转移）作为预算，RL 环境步长为 30 m。每 12 个 rollout 调度一次独立评估，共 33 个周期评估点。方法消融输出两张核心图表：图 1 为近 12 rollouts 违规率与到达率；图 2 为停站误差、时间误差、能耗（kWh）与舒适度 2×2 子图（含 300–396 rollout 放大图与阈值线）。表 1 汇总四个时期的违规率与到达率，表 2 汇总各方法 5 种子 `best/` 轨迹严格可行率与各项性能均值 ± 样本标准差，并在控制台输出代表性 PPO+PIRS 策略路径。

```bash
# 预览方法消融矩阵
uv run python -m scripts.run_method_ablation train --dry-run

# 正式方法消融
uv run python -m scripts.run_method_ablation train
```

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

# 使用安全势函数预设，并附加实验标签
python -m scripts.train_rl --run-mode monitor_best --reward-preset basic_safety --experiment-tag exp_a

# 仅预览 monitor_best 训练配置与输出路径
python -m scripts.train_rl --run-mode monitor_best --reward-preset basic_safety --dry-run

# 低开销训练，仅保留 best-eval
python -m scripts.train_rl --run-mode best_only

# 430s tune，每 12 个 rollouts 触发一次 best-eval
python -m scripts.train_rl --output-root output/optimal/rl/ --schedule-time-s 430.0 --step-distance 100.0 --run-mode tune --training-episodes 5000 --num-envs 8 --evaluation-interval-rollouts 12 --evaluation-deterministic --device cpu

# 430s monitor_best，每 6 个 rollouts 评估一次
python -m scripts.train_rl --output-root output/optimal/rl/safety_speed/ --schedule-time-s 430.0 --step-distance 100.0 --run-mode monitor_best --training-episodes 5000 --num-envs 8 --evaluation-interval-rollouts 6 --evaluation-deterministic --device cpu
```

---

### RL 评估 · `evaluate_rl`

加载用户明确指定的 PPO 模型目录，在单环境中执行评估 rollout，可选录制视频。目录必须直接包含 `policy.zip` 与 `metadata.json`，脚本不会搜索或推断 `best` / `final`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-dir` | `str` | 必填 | 直接包含 `policy.zip` 与 `metadata.json` 的模型目录 |
| `--reward-discount` | `float` | 从 `metadata.json` 读取 | 折扣因子（重建环境用） |
| `--schedule-time-s` | `float` | 从 `metadata.json` 读取 | 规划运行时间 |
| `--step-distance` | `float` | 从 `metadata.json` 读取 | 环境固定空间控制步长 (m) |
| `--reward-preset` | `str` | 从 `metadata.json` 读取 | 评估所使用的原始尺度奖励预设 |
| `--device` | `str` | `cpu` | 推理设备 |
| `--deterministic` | `bool` | `True` | 是否使用确定性策略 |
| `--record-video` | `bool` | `False` | 是否录制评估视频 |
| `--save-trajectory` | `bool` | `True` | 是否保存轨迹 NPZ 与指标 JSON |
| `--video-folder` | `str` | `mtto_eval_video` | 视频输出目录 |
| `--output-dir` | `str` | `None` | 轨迹文件输出目录（默认使用 `--model-dir`） |
| `--video-length` | `int` | `10000` | 最大录制步数 |
| `--video-trigger-step` | `int` | `0` | 视频录制触发步数 |
| `--dry-run` | `bool` | `False` | 仅解析有效评估配置、训练元数据回填结果与输入输出路径，不加载模型或运行 rollout |

评估成功后保存统一命名的 `trajectory.npz` 与 `metrics.json`。评估元数据中的模型目录使用相对输出目录的路径。

```bash
# 录制视频
python -m scripts.evaluate_rl --model-dir output/optimal/rl/.../final/ --record-video

# 指定模型目录与设备
python -m scripts.evaluate_rl \
    --model-dir output/optimal/rl/.../final/ \
    --device cuda

# 覆盖训练元数据中的 reward preset 与时间参数
python -m scripts.evaluate_rl \
    --model-dir output/optimal/rl/.../final/ \
    --reward-preset basic_safety \
    --schedule-time-s 430.0

# 仅预览有效评估配置
python -m scripts.evaluate_rl --model-dir output/optimal/rl/.../final/ --dry-run
```

---

### RL 中途计划时间突变实验 · `run_schedule_time_change`

从一次完整的方法消融中读取 5 个 PPO+PIRS（`ppo_pirs`）种子的 `best/`
和 `final/`，对 10 个候选分别运行计划时间突变评估。根 summary 根据跨工况的
严格可行性、安全、完成、停站/准点误差和能耗选出最稳健候选；`show` 只绘制该候选。
默认工况为 `Original`、`Plus 30s`、`Minus 30s`。

#### `evaluate` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--method-ablation-dir` | `str` | 必填 | 完整方法消融的 `YYYYMMDD_NN` 目录 |
| `--output-dir` | `str` | 必填 | 新的时间突变 `YYYYMMDD_NN` 目录；已存在时拒绝覆盖 |
| `--device` | `str` | `cpu` | 推理设备 |
| `--deterministic` | `bool` | `True` | 是否使用确定性策略 |
| `--change-distance-m` | `float` | `8000.0` | 触发计划时间变化的位置 (m)，即线路 8 km 处 |
| `--delta-times-s` | `str` | `0,30,-30` | 逗号分隔的计划时间变化量；新计划时间为 `schedule_time_s + delta` |
| `--dry-run` | `bool` | `False` | 仅解析配置与路径，不加载模型或运行 rollout |

`evaluate` 不接受奖励、步长、折扣或初始计划时间覆盖，并会在输出目录中保存：

- `candidates/<run_id>__<best|final>/<case>/trajectory.npz`
- `candidates/<run_id>__<best|final>/<case>/metrics.json`
- 每个候选的 `schedule_time_change_summary.json`
- 根 `schedule_time_change_summary.json`

#### `show` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load-dir` | `str` | 必填 | 直接包含根 summary 的 `YYYYMMDD_NN` 结果目录 |
| `--save-figure` | `bool` | `True` | 是否保存对比图 |
| `--show` | `bool` | `True` | 是否弹出图窗 |
| `--factor` | `float` | `0.99` | 绘制安全防护背景时使用的安全系数 |

启用 `--save-figure` 时，图像固定保存为该实验目录下的
`schedule_time_change_comparison.pdf`，不接受文件名或扩展名覆盖。

```bash
# 仅预览将要运行的突变实验矩阵
python -m scripts.run_schedule_time_change evaluate \
    --method-ablation-dir output/paper_experiment/02_method_ablation/20260910_01 \
    --output-dir output/paper_experiment/04_schedule_time_change/20260910_01 \
    --dry-run

# 评估完整方法的 10 个 best/final 候选
python -m scripts.run_schedule_time_change evaluate \
    --method-ablation-dir output/paper_experiment/02_method_ablation/20260910_01 \
    --output-dir output/paper_experiment/04_schedule_time_change/20260910_01 \
    --change-distance-m 8000.0

# 自定义突变位置和时间变化组合
python -m scripts.run_schedule_time_change evaluate \
    --method-ablation-dir output/paper_experiment/02_method_ablation/20260910_01 \
    --output-dir output/paper_experiment/04_schedule_time_change/20260910_02 \
    --change-distance-m 12000.0 \
    --delta-times-s 0,-5,5,-15,15

# 展示指定批次的最优候选，并保存对比图
python -m scripts.run_schedule_time_change show \
    --load-dir output/paper_experiment/04_schedule_time_change/20260910_01

# 只保存图像，不弹出图窗
python -m scripts.run_schedule_time_change show \
    --load-dir output/paper_experiment/04_schedule_time_change/20260910_01 \
    --no-show
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

加载用户明确指定的模型目录中的 RL 单条轨迹及指标，叠加防护曲线背景渲染。模型目录可以是一次训练的 `best/` 或 `final/`，并须直接包含统一命名的产物。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-dir` | `str` | 必填 | 直接包含 `trajectory.npz`、`metrics.json` 和 `metadata.json` 的目录 |
| `--no-safeguard` | `flag` | — | 不绘制防护曲线背景 |
| `--factor` | `float` | `0.99` | 防护曲线渲染因子 |
| `--dry-run` | `bool` | `False` | 仅解析将加载的轨迹产物与 metrics 路径，不读取数据或显示图窗 |

```bash
python -m scripts.show_rl_result --model-dir output/optimal/rl/.../best/
python -m scripts.show_rl_result --model-dir output/optimal/rl/.../final/
python -m scripts.show_rl_result --model-dir output/optimal/rl/.../final/ --dry-run
python -m scripts.show_rl_result --model-dir output/optimal/rl/.../best/ --no-safeguard
```

---

### 三基线速度曲线对比 · `compare_speed_profiles`

在同一窗口内对比展示 DP、RL 与实际运行的速度曲线，包含三联图：
- 速度-位置轨迹叠加（含安全防护背景）
- 加速度-位置曲线
- 累计总能耗（牵引+悬浮）-位置曲线

终端会以统一评价口径输出时间误差、停站误差、总能耗（kWh）和舒适度 TAV（m/s²）的三基线对比表；指定 `--output-dir` 时额外保存 `dp_rl_actual_comparison_table.md`。实际运行曲线默认读取 `output/real_operation/aligned_real_operation_curve.npz`，其加速度估算口径不同，TAV 显示为“—”；首次使用前请先运行 `python -m scripts.transform_real_operation_curve`。

DP 轨迹仍从给定目录解析；RL 轨迹只从用户明确指定的模型目录读取。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dp-curve-dir` | `str` | `output/optimal/dp` | DP 输出根目录，递归搜索最新曲线产物 |
| `--rl-model-dir` | `str` | 必填 | 直接包含 RL 统一轨迹产物的模型目录 |
| `--real-curve` | `str` | `output/real_operation/aligned_real_operation_curve.npz` | 重标定后的实际运行曲线 NPZ |
| `--no-safeguard` | `flag` | — | 速度图不渲染 safeguard 背景 |
| `--factor` | `float` | `0.99` | safeguard 背景渲染因子 |

```bash
# 指定 DP 目录、RL 模型目录与实际运行曲线
python -m scripts.compare_speed_profiles \
    --dp-curve-dir output/optimal/dp \
    --rl-model-dir output/optimal/rl/.../best/ \
    --real-curve output/real_operation/aligned_real_operation_curve.npz

# 关闭 safeguard 背景并调整渲染因子
python -m scripts.compare_speed_profiles --rl-model-dir output/optimal/rl/.../best/ --no-safeguard --factor 0.97
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
| `--rl-model-dir` | `str` | RL 模式必填 | 直接包含 RL 统一轨迹产物的模型目录 |
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
# 默认比较模式：文本摘要 + 主图（含事件 marker）
python -m scripts.analyze_sps_compliance --rl-model-dir output/optimal/rl/.../best/

# 仅输出文本
python -m scripts.analyze_sps_compliance --rl-model-dir output/optimal/rl/.../best/ --output-mode text

# 输出 JSON 并写入文件
python -m scripts.analyze_sps_compliance \
    --rl-model-dir output/optimal/rl/.../best/ \
    --output-mode json \
    --json-output-path output/optimal/sps_compliance_report.json

# 主图启用 marker-only（不显示文本注释）
python -m scripts.analyze_sps_compliance --rl-model-dir output/optimal/rl/.../best/ --event-annotation marker-only

# 指定 RL 模型目录与 SPS 时延参数
python -m scripts.analyze_sps_compliance \
    --rl-model-dir output/optimal/rl/.../final/ \
    --step-delay-s 2.0
```

---

### 线路环境与防护曲线 · `show_env_data`

可视化展示磁浮列车运行线路环境数据与安全防护曲线，支持 `--view {overview,full-curves,danger-region,all}` 视图切换：
- `overview`（默认）：综合环境视图（上图为防护曲线+车站/加速区/ASA，下图为轨道坡度）。
- `full-curves`：全量安全防护曲线视图（Safe levitation, safe braking, min/max curves 与基础设施）。
- `danger-region`：危险交叉点与局部危险速度域视图。
- `all`：同时生成并展示以上所有视图。

```bash
# 默认展示综合环境视图（含坡度）
python -m scripts.show_env_data

# 展示全量安全防护曲线
python -m scripts.show_env_data --view full-curves

# 展示危险速度域与交叉点
python -m scripts.show_env_data --view danger-region

# 保存为图片且不弹出交互窗口
python -m scripts.show_env_data --view overview --output-dir output/figures --no-show
```

---

### 计算并保存防护曲线 · `calc_and_save_safeguard_curves`

离线计算并序列化保存完整的安全防护曲线数据至
`output/safeguardcurves/`，供后续训练与评估加载。默认拒绝覆盖已有产物；
确认重新生成时需要指定 `--force`。

```bash
python -m scripts.calc_and_save_safeguard_curves --dry-run
python -m scripts.calc_and_save_safeguard_curves --force
```

可以调整距离步长、车辆参数、安全误差与执行延时，并将结果保存到自定义目录：

```bash
python -m scripts.calc_and_save_safeguard_curves \
  --output-dir output/safeguardcurves_0p5m \
  --distance-step-m 0.5
```

查看完整参数：

```bash
python -m scripts.calc_and_save_safeguard_curves --help
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

可视化训练奖励实际使用的安全势函数和准点势函数。单面板图采用 85 mm 的
SCI 单栏宽度，多面板联合图采用 170 mm 的双栏宽度。所有静态图按固定物理尺寸
保存为 PDF，嵌入栅格元素使用 1200 DPI。用户只指定 `--output-dir`，文件名和
`.pdf` 扩展名由脚本根据图类型固定。
项目图片的英文与数学字形统一优先使用 Arial；Arial 不可用时回退到
Liberation Sans 等兼容无衬线字体，并以 TrueType 形式嵌入 PDF；中文字形使用
Noto Sans CJK 无衬线回退。
脚本仅保留运行时实际使用的速度安全势与准点势，不再提供位置安全势或停站势。

```bash
# 完整线路上的位置–冗余运行时间准点势热图
python -m scripts.show_potential_function --plot-type punctuality

# 安全势与准点势双栏并排展示
python -m scripts.show_potential_function --plot-type safety-punctuality

# 覆盖准点势使用的计划运行时间
python -m scripts.show_potential_function \
  --plot-type punctuality --schedule-time-s 465
```

---

### 终端评分函数可视化 · `show_score_function`

可视化展示停站精度评分函数 $f_s(x)$ 与准点率评分函数 $f_p(t)$（支持 `--plot {combined,stopping,punctuality}`）：
- `combined`（默认）：停站与准点评分函数综合双子图。
- `stopping`：停站误差评分曲线与死区/衰减阈值。
- `punctuality`：准点时间误差评分衰减曲线。

```bash
# 展示停站与准点综合评分曲线
python -m scripts.show_score_function --plot combined

# 展示停站误差评分曲线
python -m scripts.show_score_function --plot stopping

# 展示准点时间误差评分曲线
python -m scripts.show_score_function --plot punctuality

# 保存为图片且不弹出交互窗口
python -m scripts.show_score_function --plot combined --output-dir output/figures --no-show
```

---

## 测试与代码检查

```bash
# 全量测试
PYTHONPATH=. uv run pytest -q

# 指定测试文件
PYTHONPATH=. uv run pytest -q tests/test_mtto_env.py

# 代码风格与静态检查
uv run ruff check .
```
