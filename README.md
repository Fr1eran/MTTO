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
│   ├── context_pool.py     #   DP 参考轨迹与不可变上下文池构建
│   ├── context_sampler.py  #   持有版本化分布的上下文采样器
│   ├── dspdl.py            #   DSPDL 共享统计与课程回调
│   ├── dspdl_distribution.py      # DSPDL 通用分布求解器
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
| 方法消融与代表策略选择 | `python -m scripts.run_method_ablation train/show` |
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

#### DSPDL 课程学习

主训练入口默认使用 `--curriculum-profile dspdl`。课程启用时仍需通过
`--reference-curve-dir <dp-output-dir>` 提供与任务匹配的 DP 参考轨迹；显式指定
`--curriculum-profile none` 可关闭课程。旧任务完成度课程的输出目录和 manifest 仅作为
历史材料保存，不会被当前脚本自动迁移或续跑。

父进程读取 `ReferenceTrajectory`，由 `ContextPoolBuilder` 对 DP 轨迹采样并重建为只读
`ContextPool`，再将同一逻辑任务池交给所有训练环境。每个训练环境只创建轻量的
`ContextSampler`，不会重复读取或重建 DP 轨迹。

DSPDL 只对当前课程更新窗口中实际采样的唯一上下文计算 PPO
`predict_values()`，并按论文 Eq. (5) 构造 DSPDL 系数：

$$
g(c)=\frac{n_c}{Kp_i(c)}V_\theta(c)\quad(n_c>0),\qquad g(c)=0\quad(n_c=0),
$$

其中 $K$ 是窗口中的 episode-start 样本总数、$n_c$ 是上下文 $c$ 的采样次数、
$p_i(c)$ 是该窗口对应的课程分布。环境按原始奖励累计
$G=\sum_{t=0}^{T-1}\gamma^t r_t$ 用于课程强度和价值校准诊断，
不执行 ZPD 转换、奖励缩放或 VecNormalize。

训练时可在 TensorBoard 查看 DSPDL 的 `dspdl/alpha`、KL、采样与校准诊断。
协议固定为 `zeta=4`、相对熵约束 `0.01`、每 4 个 rollout 更新且 warmup 4 次，
不提供运行时调参或替代估计器入口。

#### 奖励配置与实验标识

默认奖励预设为 `basic_safety_punctuality`：在 `basic_safety` 上加入线性剩余裕度准点
PBRS。无需 DP 参考即可启用该奖励；启用课程时仍按课程要求提供 DP 数据。若要复现
不含准点势函数的旧设置，可显式传入 `--reward-preset basic_safety`。

令 `q` 为沿运行方向计算并裁剪到 `[0,1]` 的剩余距离比例，`b0` 为全程静止起点的
计划时间减最短运行时间，参考裕度为 `b_ref=b0*q`。课程中途起点沿用全程参考线。
实际裕度与参考的差为 `e`，势函数为 `Phi=-K*e²/(sigma²+e²)`，新增奖励为
`gamma*Phi(next)-Phi(previous)`。参数固定为 `K=5`、`sigma=20 s`，不提供运行时覆盖。
负初始裕度保留符号；势函数不替代原终端准点评分。

新预设在内部任务结束（包括失败和步数上限）将准点终端势归零，并关闭该结束状态的
时间限制 bootstrap。PPO rollout 边界不归零；外部采样时间限制保留 bootstrap。
DSPDL 使用 `V_shaped+Phi(start)`，课程回报统计剔除新增准点塑形分量，
保持原奖励比较口径。旧安全 PBRS 的边界行为保持不变。
计划时间变化后以新计划更新全程参考线，不从变化点重锚；变更接口本身不发奖励，
不对跨外部计划变更的整段回报宣称固定任务策略不变性。

```bash
python -m scripts.train_rl --reward-preset basic_safety_punctuality --curriculum-profile none
python -m scripts.show_potential_function --plot-type punctuality-slack --no-show --output-file output/punctuality_slack.png
```

诊断中新增 `punctuality_shaping` 分量，诊断 schema 为 4，读取旧版 2/3 时该分量补零。
上述参数是实验起点，短程训练验证不代表准点性能提升。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--reward-preset` | `str` | `basic_safety_punctuality` | 原始尺度奖励预设；默认启用准点势函数 |
| `--experiment-tag` | `str` | `None` | 附加实验标签，用于隔离输出目录与 TensorBoard 运行名 |

课程配置参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--curriculum-profile` | `str` | `dspdl` | `none`、`dspdl` |
| `--reference-curve-dir` | `str` | `None` | 启用课程时必填，指向与任务匹配的 DP 轨迹目录 |

`basic` 固定包含 `energy + comfort`，`basic_safety` 在此基础上启用安全 PBRS；
默认的 `basic_safety_punctuality` 进一步启用准点 PBRS。训练和评估均直接使用原始
奖励尺度，停站精度和准点终端评分函数保持不变。

#### PPO 超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--reward-discount` | `float` | `0.995` | 回报折扣因子 γ |
| `--rollout-steps-per-update` | `int` | `2048` | 每次更新的 rollout 总步数 |
| `--n-steps-per-env` | `int` | 自动推导 | 每个环境的步数（优先级高于 `--rollout-steps-per-update`） |
| `--training-episodes` | `int` | `7000` | 全局完成训练回合数；按环境数取整并自动推导 PPO 环境步上限 |
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

每次训练会写入版本化 `metadata.json`，并在 `final/` 下生成 `episodes.npz`。该二进制产物保存完整 episode 奖励分量累计值及 rollout 级 transition 充分统计量；奖励占比与相关性由训练后分析模块计算，不再写入 TensorBoard event。

#### Best-Eval（训练期最优轨迹评估）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-best-evaluation-artifacts` | `bool` | 取决于 run-mode | 启用训练期最优模型与轨迹产物 |
| `--evaluation-interval-rollouts` | `int` | `12` | 每完成指定数量的 PPO rollouts 执行一次评估 |
| `--evaluation-deterministic` | `bool` | `True` | 是否使用确定性策略推理 |
| `--evaluation-history-path` | `str` | `None` | 可选的完整评估历史 NPZ 输出路径 |

评估仅在 rollout 边界触发，不再在每个训练 step 中检查，也不再支持按完成 episode 数调度。评估历史使用 `rollout_indices` 与 `training_steps` 同时标记评估时点；最优轨迹 metrics 使用 `evaluation_rollout_index`。

成功判定使用 `TrainService.max_stop_error` 与 `TrainService.max_arr_time_error_s`：
- `stop_error_m <= max_stop_error`
- `abs(time_error_s) < max_arr_time_error_s`，默认准点阈值为 `10.0s`

Best-eval 排序规则：
- 一旦出现成功轨迹，所有成功轨迹都优先于所有未成功轨迹
- 在成功轨迹之间，优先比较总能耗，越小越优
- 如果当前还没有成功轨迹，才回退到按总 reward 比较
- 停站误差与绝对时间误差仅作为稳定 tie-break

每次刷新最优时，在实验目录下的 `best_rollouts/` 中保存模型、`best_trajectory.npz` 与版本化 `metrics_best.json`。

论文方法消融将 PBRS 作为安全势函数与准点势函数的整体开关，不拆分两个分量。

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
python -m scripts.train_rl --run-mode tune \
  --curriculum-profile dspdl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0

# 高效复现（关闭日志，不进行 best model 评估，仅得到最终训练模型）
python -m scripts.train_rl --run-mode reproduce \
  --curriculum-profile dspdl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0

# 关闭高频回调，保留基础监控 + best-eval
python -m scripts.train_rl --run-mode monitor_best \
  --curriculum-profile dspdl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0

# 使用安全 PBRS 预设，并附加实验标签
python -m scripts.train_rl --run-mode monitor_best --reward-preset basic_safety --experiment-tag exp_a \
  --curriculum-profile dspdl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0

# 仅预览 monitor_best 训练配置与输出路径
python -m scripts.train_rl --run-mode monitor_best --reward-preset basic_safety \
  --curriculum-profile dspdl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0 --dry-run

# 低开销训练，仅保留 best-eval
python -m scripts.train_rl --run-mode best_only \
  --curriculum-profile dspdl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0

# 430s tune，每 12 个 rollouts 触发一次 best-eval
python -m scripts.train_rl --output-root output/optimal/rl/ --schedule-time-s 430.0 --step-distance 100.0 --curriculum-profile dspdl --reference-curve-dir output/optimal/dp/ --run-mode tune --training-episodes 7000 --num-envs 8 --evaluation-interval-rollouts 12 --evaluation-deterministic --device cpu

# 430s monitor_best，每 6 个 rollouts 评估一次
python -m scripts.train_rl --output-root output/optimal/rl/safety_speed/ --schedule-time-s 430.0 --step-distance 100.0 --curriculum-profile dspdl --reference-curve-dir output/optimal/dp/ --run-mode monitor_best --training-episodes 7000 --num-envs 8 --evaluation-interval-rollouts 6 --evaluation-deterministic --device cpu
```

---

### RL 评估 · `evaluate_rl`

加载训练好的 PPO 模型，在单环境中执行评估 rollout，可选录制视频。若 `--load-dir` 所在实验目录存在 `metadata.json`，评估会优先复用其中的 `schedule_time_s`、`reward_discount`、`step_distance` 与 `reward_preset`；只有在显式传参时才覆盖这些值。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load-dir` | `str` | `output/optimal/rl/final/` | PPO 模型所在目录 |
| `--reward-discount` | `float` | 从 `run_metadata.json` 读取，否则 `0.995` | 折扣因子（重建环境用） |
| `--schedule-time-s` | `float` | 从 `run_metadata.json` 读取，否则 `430.0` | 规划运行时间 |
| `--step-distance` | `float` | 从 `run_metadata.json` 读取，否则 `30.0` | 环境固定空间控制步长 (m) |
| `--reward-preset` | `str` | 从 `run_metadata.json` 读取，否则 `basic_safety` | 评估所使用的原始尺度奖励预设 |
| `--device` | `str` | `cpu` | 推理设备 |
| `--deterministic` | `bool` | `True` | 是否使用确定性策略 |
| `--record-video` | `bool` | `False` | 是否录制评估视频 |
| `--save-trajectory` | `bool` | `True` | 是否保存轨迹 NPZ 与指标 JSON |
| `--video-folder` | `str` | `mtto_eval_video` | 视频输出目录 |
| `--output-dir` | `str` | `None` | 轨迹文件输出目录（默认回退到 `--load-dir`） |
| `--video-length` | `int` | `10000` | 最大录制步数 |
| `--video-trigger-step` | `int` | `0` | 视频录制触发步数 |
| `--dry-run` | `bool` | `False` | 仅解析有效评估配置、训练元数据回填结果与输入输出路径，不加载模型或运行 rollout |

评估成功后可保存 `final_trajectory.npz` 与版本化 `metrics_final.json`。最终轨迹指标会显式写入 `trajectory_source=final`，并与训练期的最优轨迹共用同一套 selection metadata 结构。评估指标统一以 J 保存能耗，kJ 仅用于展示换算。

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

实验 4 默认批量运行 `Original`、`Plus 30s`、`Minus 30s` 三种情形。其中 `Original` 不触发计划时间变化，其余情形会在列车首次跨过 `--change-distance-m` 指定位置时调用环境的 `change_schedule_time()`。

#### `evaluate` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load-dir` | `str` | `output/optimal/rl/final/` | PPO 模型所在目录 |
| `--output-dir` | `str` | `output/optimal/rl/schedule_time_change_eval/` | 突变实验输出根目录；每次运行会创建时间戳子目录 |
| `--reward-discount` | `float` | 从 `run_metadata.json` 读取，否则 `0.995` | 折扣因子（重建环境用） |
| `--schedule-time-s` | `float` | 从 `run_metadata.json` 读取，否则 `430.0` | 突变前的初始规划运行时间 |
| `--step-distance` | `float` | 从 `run_metadata.json` 读取，否则 `30.0` | 环境固定空间控制步长 (m) |
| `--reward-preset` | `str` | 从 `run_metadata.json` 读取，否则 `basic_safety` | 评估所使用的原始尺度奖励预设 |
| `--device` | `str` | `cpu` | 推理设备 |
| `--deterministic` | `bool` | `True` | 是否使用确定性策略 |
| `--change-distance-m` | `float` | `800.0` | 触发计划时间变化的位置 (m) |
| `--delta-times-s` | `str` | `0,30,-30` | 逗号分隔的计划时间变化量；新计划时间为 `schedule_time_s + delta` |
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

# 使用训练得到的 final 模型运行默认三组突变实验
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
python -m scripts.show_env_data --view overview --output-file output/figures/env_overview.png --no-show
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

可视化安全势函数（Safety Speed / Safety Position / Safety Speed Adaptive），用于 RL reward shaping。

```bash
python -m scripts.show_potential_function
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
python -m scripts.show_score_function --plot combined --output-file output/figures/score_functions.png --no-show
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
