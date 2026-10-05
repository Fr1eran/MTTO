# 高速磁浮速度曲线优化：论文仿真实验指南

本手册描述论文全套仿真实验：控制周期步长消融、方法消融、DP/PIRS/实际运行三方对比、途中计划时分变更。核心强化学习方法称为物理先验奖励塑形（Physics-Informed Reward Shaping, PIRS），内部标识为 `ppo_pirs`。所有命令均从项目根目录执行，统一通过 `uv run` 使用项目虚拟环境；论文相关依赖（matplotlib、pandas、openpyxl）随 `dev` 依赖组一并安装（见 `uv sync`）。

实验编排代码位于 `paper/experiments/`（`spec.py` 解析 `paper/specs/*.toml` 并展开实验矩阵，`runner.py` 负责调用 `mtto.workflows` 并处理中断恢复与结果复用），命令行入口为 `python -m paper.experiments`。出图脚本位于 `paper/figures/`。第 4 节描述结果复用规则；不熟悉该机制时，重新运行下列命令是安全的——已完成且未过期的结果会被自动复用而不是重新计算。

## 1. 实验结构与固定口径

完整仿真实验分为四部分：

1. 控制周期步长消融（第 5 节）：确定后续实验使用的控制周期 `step_time_s`。
2. 方法消融（第 6 节）：对比 PPO、PPO+Safety、PPO+Punctuality 和 PPO+PIRS。
3. DP、PIRS 与实际运行结果对比（第 7 节）。
4. 途中计划时分变更：PIRS 代表策略推理与 DP 重新求解的轨迹指标和重新计算耗时对比（第 8 节）。

| 项目 | 固定值 | 定义位置 |
| --- | --- | --- |
| 场景 | 上海高速磁浮示范线下行场景（龙阳路 → 浦东国际机场） | `paper/specs/scenario.toml`、`paper/specs/tasks.toml` 的 `longyang_to_airport` |
| 名义计划运行时间 | `465 s` | `paper/specs/tasks.toml` |
| 控制周期候选 | `0.5, 1.0, 1.5, 2.0 s` | `paper/specs/step_time.toml` 的 `[[variants]]` |
| 随机种子 | `11, 131, 239, 359, 443` | `paper/specs/method_ablation.toml`、`step_time.toml` 的 `seeds` |
| 步长/方法消融预算 | 每组 `400` 个 PPO rollout，即 `num_envs(8) * n_steps_per_env(1024) * 400 = 3,276,800` 个环境状态转移 | `[train]` 表的 `training_rollouts/num_envs/n_steps_per_env` |
| 周期评估间隔 | 每 `12` 个 rollout 一次确定性独立评估轨迹，rollout 12–396（共 33 个点） | `[train]` 表的 `evaluation_interval_rollouts` |
| 精确停站 | `abs(stop_error_m) <= max_stop_error_m`（`0.3 m`） | `paper/specs/tasks.toml`、`evaluation/quality.py` |
| 准点 | `abs(time_error_s) < max_arr_time_error_s`（`10 s`） | `paper/specs/tasks.toml`、`evaluation/quality.py` |
| 安全 | 速度曲线在动态双限速度防护下逐节点零越界（`QualityReport.safe`） | `evaluation/quality.py` |
| 论文能耗单位 | `kWh` | 图表与表格换算自 `QualityMetrics` 的 kJ |

底层物理计算与持久化产物（`profile.npz`、`quality.json`、`result.json`）以 kJ 为规范单位（能量）、m/m/s/s 为规范单位（位置/速度/时间）。论文图、性能表和终端汇总统一换算为 kWh：

```text
1 kWh = 3,600 kJ = 3,600,000 J
```

## 2. PIRS 与最好结果定义

完整 PIRS 预设为 `basic_safety_punctuality`，由速度安全势与线性剩余裕度准点势组成（`mtto.rl.rewards`）。四种方法消融变体：

| 变体 | 奖励预设 | 论文名称 |
| --- | --- | --- |
| `ppo` | `basic` | PPO |
| `ppo_safety` | `basic_safety` | PPO+Safety |
| `ppo_punctuality` | `basic_punctuality` | PPO+Punctuality |
| `ppo_pirs` | `basic_safety_punctuality` | PPO+PIRS |

势函数塑形统一按 `gamma*Phi(next) - Phi(previous)` 计算，终止步继续使用观测到的下一状态势能（不显式归零）：

- **安全势**：按制动（牵引）储备的二次合页，`Phi = -c_up*(1 - clip(k_up,0,H)/H)² - c_low*(1 - clip(k_low,0,H)/H)²`，`H=3`（`SAFETY_RESERVE_HORIZON_STEPS`）、`c_up=3`、`c_low=1`（`SAFETY_RESERVE_UPPER_SCALE`/`SAFETY_RESERVE_LOWER_SCALE`）。储备按控制周期计量：以当前速度再运行多少个控制周期，就会耗尽全力制动（牵引）的距离余量。制动储备 `k_up = min((v_max² - v²)/(2*b*ds_eff), (v_max'² - v²)/(2*b*ds_eff) + 1)`，`ds_eff = v*dt + ½*b*dt²`（二次项只用于避免低速时分母趋零），其中 `dt = step_time_s`、`b` 为最大制动减速度、`v_max'` 为一个控制周期内最远可达位置 `x + v*dt + ½*a_max*dt²` 处的上限，停车时为 +inf；牵引储备 `k_low` 同理（用 `a_max` 代替 `b`），只对大于 0 的下限计入。`k <= 0` 表示已越界或下一个控制周期无论如何都将越界；储备 `>= H` 时势能为 0（`mtto.rl.rewards.braking_reserve_steps`/`traction_reserve_steps`/`safety_potential`）。
- **准点势**：伪 Huber 形式 `Phi = -2K * (sqrt(e² + sigma²) - sigma) / sigma`，零附近为 `-K*e²/sigma²`，远离零时线性、每偏离 1 s 代价 `2K/sigma = 0.5`，不饱和，使落后时刻表的每一秒都有代价；`K=5`（`PUNCTUALITY_POTENTIAL_SCALE`）、`sigma=20 s`（`PUNCTUALITY_POTENTIAL_SIGMA_S`）；`e` 为实际剩余裕度与参考裕度之差，参考裕度按沿运行方向裁剪到 `[0,1]` 的剩余距离比例 `q` 线性插值：`b_ref = b0 * q`，`b0` 为全程静止起点的计划时间减最短运行时间。计划时间变化后以新计划更新全程参考线，不从变化点重锚。
- **逐步奖励**（记 `P = PROGRESS_REWARD_SCALE = 50`、`L` 为任务里程、`d = |x_target - x|` 为到目标点的距离、`Δs` 为本步位移、`dt = step_time_s`）：进度为主导项，能耗与舒适为按距离计量的惩罚。
  - 进度（带符号）：`P*(d - d')/L`。朝目标运行为正，越过目标点后继续前进时 `d` 增大，进度为负；从起点到目标点累计为 `P`。成功停站这一步照常计入，失败终止步不计入。
  - 能耗（只计牵引）：`-0.4*P*ΔE_prop/(e_peak*L)`（`ENERGY_REWARD_WEIGHT = 0.4`），`e_peak` 为 `RewardNormalization.peak_propulsion_kj_per_m`，由 `workflows.train.build_env_references` 数值求得：在任务区间内出现的每个坡度值上、对 `v ∈ [0.5, v_max]`（步长 0.5 m/s）以 `a_max` 运行 0.1 s，取牵引能耗除以位移的最大值（本线路约 433 kJ/m，出现在 `v ≈ 139 m/s`、`a = 1`；从静止起步的 0.1 s 步只走 5 mm，其每米损耗是离散化假象，不计入网格）。因此任意一步都有 `ΔE_prop <= e_peak*Δs`，即能耗项不低于 `-0.4*P*Δs/L`（由 `tests/rl/test_reward_properties.py` 在 `dt ∈ {0.5, 1, 2}` 的真实步上验证）。**悬浮能耗不进入奖励**：它与时间成正比，低速时每米悬浮能耗发散（`v = 1 m/s` 时约 213–320 kJ/m），会破坏“每步为正”，且准点到达时几乎与策略无关；状态与评价指标中的悬浮能耗照常累计。
  - 舒适：`-W*|Δa|*(1+ρ²)`（`COMFORT_REWARD_WEIGHT = W = 0.2`），`ρ = |Δa|/dt/j_max`，`j_max` 为 `Task.max_jerk_mps3`（`0.75 m/s³`），`Δa/dt` 按名义控制周期计冲击率。按每 m/s² 加速度变化计价，一次行程的舒适代价为 `W` 乘总变差 TAV，与控制周期和变化发生的位置无关；乘子 `(1+ρ²)` 使冲击率达阈值时单位变化的代价翻倍、超过阈值后三次增长，鼓励把大的变化拆成多个小变化。`W=0.2` 取在能耗换舒适的交换率分界（约 0.19 奖励/TAV 单位）附近：`log_std_init=-1` 下 10 个种子（两个独立种子集）TAV 由 8.19 降到 4.93，能耗 805→800 kWh（p=0.63），20/20 严格可行；`W=0.4` 时 TAV 4.03 但能耗 +32 kWh，`W=0.8` 出现失败种子。去掉 `(1+ρ²)` 的纯 `|Δa|`（`W=0.2`）TAV 为 6.18，说明该乘子确有作用。
  - 因此非终止的朝目标一步满足 `进度 + 能耗 >= 0.6*进度 > 0`：能耗永远不会抵消进度；舒适项与进度无关，不受此约束，加速度变化大的一步可以净亏。
- **终止步奖励**（仅 `STOPPED_IN_ZONE` 终止时触发）：停站分 `stopping_score(e_x) = 1 / (1 + (|e_x| / max_stop_error_m)⁴)`（`STOPPING_SCORE_POWER = 4`：处处光滑、单调，目标点处为 1，容差处恰为 1/2，远端按幂律衰减）；准点分 `punctuality_score(e_t) = exp(-|e_t| / 15)`；终端奖励 `terminal_stopping = 100 * 0.3 * stopping_score`，`terminal_punctuality = 100 * 0.7 * stopping_score * punctuality_score`（`TERMINAL_REWARD_SCALE = 100`），合计 `100*S_stop*(0.3 + 0.7*S_punc)`（`mtto.rl.rewards.RewardCalculator`）。
- **截断惩罚**：所有失败终止（`UNDER_LOWER_LIMIT`、`OVER_UPPER_LIMIT`、`OVER_SRTSP`、`STOPPED_SHORT`、`OVERRAN`）的该步只返回常数 `TRUNCATION_PENALTY = -80`（大于进度总量 `P`）加安全势与准点势的塑形项。

环境每步以指令加速度匀变速推进一个控制周期 `step_time_s`（`mtto.domain.kinematics.run_time`），步内停车时截到停车时刻，时间只累加实际时长；不对到达目标点的那一步做截短，精确停车由连续动作给出（观测 o7 即恰好停在目标点所需的加速度）。

停站判定（`mtto.domain.scenario.Task.stop_state`）：停车（`|v| <= 0.01 m/s`）且 `|x - x_target| <= 30*max_stop_error_m` 为 `STOPPED_IN_ZONE`；列车带速到达目标点时不终止，继续越过目标点，越过距离超过 `30*max_stop_error_m` 即为 `OVERRAN`；其余停车为 `STOPPED_SHORT`。控制周期不超过 1.5 s 时，带速越过目标点的列车总会先被 SRTSP 上限曲线的尾段（在目标点后约 7 m 处降到 0）以 `OVER_SRTSP` 终止，来不及越过 9 m 的停车区，因此 `OVERRAN` 只在更长的控制周期下出现。结束原因的优先级为停站判定 → 控制周期末的速度越限。回合不设截止时刻：晚点只由不饱和的准点势与终端准点评分约束，严格可行（`|e_t| <= max_arr_time_error_s`）仍在质量评估中判定。按截止时刻截断并自举时，慢速运行到截止会成为无惩罚的出口；按失败终止时，“太慢”与“快但进站失败”同为 −80，二者都妨碍学习。

智能体观测为 13 维（`mtto.rl.observation.ObservationBuilder`，`POLICY_IO_VERSION=3`）：o0 全程进度、o1 带符号对数距离（尺度 0.1 m）、o2 速度、o3 上一步加速度、o4/o5 速度上/下限、o6 准点比值 `e/sqrt(e²+sigma²)`、o7 恰好停在目标点所需的加速度、o8 制动储备 `clip(k_up,0,H)/H`、o9 冗余时间对数编码（尺度 10 s）、o10 全力制动下的预计停站误差 `v²/(2b) - d` 的对数编码（尺度 0.1 m）、o11 坡度 `clip(i/i_max,-1,1)`（上坡为正，`i_max` 为车辆爬坡能力 `max_slope_capacity`）、o12 牵引储备 `clip(k_low,0,H)/H`。速度按线路最高限速归一化。

安全越界判定只在 `mtto.domain.safeguard`（动态双限速度防护，按节点）；算法的运行上限（SRTSP、静态允许速度域）属于算法约束，不进入质量判定。RL 的 SRTSP 上限曲线自轨道原点 0 m 静止起算至目标点后 `20*max_stop_error_m`，任务起点静止出发的列车始终在其下方；查找表间隔 1 m，格内按 `v²` 线性插值（匀加速与制动段上精确）。DP 使用实际任务起点至实际目标点、首尾速度均为零的最短运行时间曲线作为可达速度上界。所有方法的环境内部终止均视为真实任务结束，映射为 `terminated`，失败终点不参与 PPO 的 bootstrap；只有外部时间上限映射为 `truncated=True`。

质量评估（`mtto.evaluation.quality.assess`）的舒适度指标：TAV 为逐段加速度变化量 `|Δa|` 之和，RMS 为其均方根，ER% 为逐段冲击率 `|Δa|/Δt` 超过 `max_jerk_mps3` 的段数占比（`Δt = 0` 的段不计入分子与分母）。

“最好结果”按以下口径执行（`mtto.evaluation.quality.selection_key`/`best_update_reason`，由 `ScheduledPolicyEvaluationCallback` 在训练期周期评估中维护）：

- 消融性能表与控制周期表对每个随机种子同时报告训练期 `best/` 检查点与训练结束时的 final 策略（运行目录顶层），两行并列。
- 择优优先级：严格可行（安全、成功、精确停站且准点）轨迹优先于所有不可行轨迹；严格可行轨迹之间优先选择总能耗更低者；尚无严格可行轨迹时，依次按“安全且成功”、停站精度、准点性、时间误差绝对值和能耗回退。
- 消融学习曲线使用全部 5 个随机种子：先对单种子做 5 点尾随滑动平均，再计算多种子均值与样本标准差带（ddof=1）。
- 消融性能表对 5 个种子的 best 与 final 评估指标分别报告均值 ± 样本标准差（ddof=1），包含全部 5 个种子，不挑选单个种子的最优数值。
- 代表策略取各种子 final 策略中严格可行且能耗最低者（无可行者按 `selection_key` 回退），不用 `best/`：best 是可行评估中能耗最低者，倾向于在时分容限内晚到。三方轨迹对比与途中时分变更始终使用 PPO+PIRS 的代表策略。

## 3. 命令入口一览

| 用途 | 命令 |
| --- | --- |
| 单次 RL 训练/评估/DP/训练分析 | `uv run mtto {train,evaluate,dp,analyze-training} ...`（见项目根 `README.md`） |
| 步长消融（训练矩阵，只出表） | `uv run python -m paper.experiments step_time {run,summarize} [--spec ...] [--workers N] [--output ...]` |
| 方法消融（训练矩阵） | `uv run python -m paper.experiments method_ablation {run,summarize,figures} [--spec ...] [--workers N] [--output ...]` |
| 途中计划时分变更（评估 + DP 重新求解） | `uv run python -m paper.experiments schedule_change {run,summarize,figures} [--spec ...] [--output ...]` |
| 实测运营数据重标定 | `uv run python -m paper.real_operation [--output-file ...]` |
| 环境、势函数、评分函数说明图 | `uv run python -m paper.figures.{env_data,potential_function,score_function}` |
| DP/RL 单条轨迹可视化 | `uv run python -m paper.figures.{dp_result,rl_result} --dp-run/--rl-run <目录>` |
| 三方（DP/PIRS/实际）对比与 DP 冗余时间误差、SPS 合规分析 | `uv run python -m paper.figures.{speed_profile_comparison,dp_redundancy_error,sps_compliance}` |
| 最短运行时间曲线、实测数据曲线（无参数，交互式脚本） | `uv run python -m paper.figures.{min_operation_time_curve,real_operation_data}` |

所有出图脚本支持 `--no-show`（保存不弹窗）与大多数支持 `--output-dir`（固定文件名，见各脚本 `--help`）；`min_operation_time_curve`（交互式最短运行时间曲线计算器，键盘输入起点重新计算）与 `real_operation_data`（打印实测曲线统计并展示三张图）没有命令行参数，始终调用 `plt.show()`；在无显示环境下用 `MPLBACKEND=Agg` 运行会跳过弹窗但仍完整执行其余逻辑。

`run`/`summarize`/`figures` 的 `--spec` 默认分别为 `paper/specs/step_time.toml`、`paper/specs/method_ablation.toml`、`paper/specs/schedule_change.toml`；`run` 不需要 `--output`（训练/评估产物写入 spec 的 `output_root`），`summarize`/`figures` 需要 `--output <目录>`（汇总 JSON、Markdown 表格、PDF 图表写入该目录，可以是任意路径，不必与 `output_root` 相同）。

**训练矩阵的并行执行**：`step_time run` 与 `method_ablation run` 把需要训练的矩阵单元交给一个固定大小的进程池（`--workers N`，默认 CPU 核数 − 2），空闲进程依次领取下一个单元，进程数与种子数、变体数无关；`--workers 1` 在当前进程内依次训练。每次训练只用一个 torch 线程（`mtto.rl.ppo.TORCH_NUM_THREADS`），结果与由哪个进程、在何时训练无关，并行与依次执行的产物逐位一致；`--workers` 不进入 `reuse_key`。中断的残留目录、编号分配与 `paper.json` 都在主进程中处理（第 4 节规则不变）。不要同时启动多个实验命令，也不要在训练时并行运行 DP 等其他多进程任务。

## 4. 结果复用与 `paper.json`

`paper/experiments/runner.py` 在每个训练/评估运行目录中，紧邻 `mtto` 产物之外另写 `paper.json`：

```text
reuse_key = sha256(确定性编码的 {run.json 中的 config、scenario_hash、task、mtto_version，
                                  以及评估运行所评估策略的 policy.zip 的 SHA-256})
git_commit, dirty   # 运行开始时的溯源信息
```

- **完成判定**：`run.json` 存在、该 `kind` 的必需产物齐全且能被 `mtto.io.artifacts` 严格读取、按当前产物重新计算的 `reuse_key` 与 `paper.json` 一致；任一不满足即视为中断残留，运行前自动清理。
- **训练预算未达成**：若 `result.json` 的 `training.target_reached` 为假（例如被派生时间步上限截断），训练是确定的、重跑结果相同，因此整个实验矩阵在此处停止（`ExperimentStopped`），不重跑，不跳过。
- **复用规则**（自动进行，无需传参）：已完成且 `reuse_key` 相同 → 复用，但当前工作区干净（`git status --porcelain` 为空）时不复用 `dirty=true` 的旧结果；`reuse_key` 不同，或因上一条被拒绝复用 → 保留原目录，新运行写入新的 `__NN` 后缀目录与 `run_id`，不删除已有结果。
- **`figures` 拒绝脏结果**：`summarize`/`figures` 读取的每个运行目录，其 `paper.json.dirty` 为真时，`figures` 会直接报错（`require_clean`）；`summarize` 不做此检查。因此正式论文图表必须在干净工作区下生成（可先 `summarize` 检查数值，工作区干净后再 `figures` 出图）。
- **计算逻辑变化的识别**：不做代码哈希。修改计算逻辑（物理、判定或奖励相关）时必须同步递增 `pyproject.toml` 的 `project.version`，`reuse_key` 随之改变，旧结果不再被复用；只修改 `paper/`、CLI 或文档时不递增，结果继续复用。

`mtto dp` 是独立的一次性运行（`FileExistsError` 拒绝覆盖已存在的输出目录），不经过 `paper/experiments` 的复用层；第 7 节的 DP 运行需要人工为每次实验选择新的输出目录。

## 5. 控制周期步长消融（论文实验一）

比较 `step_time_s = 0.5/1.0/1.5/2.0 s × 5 seeds`，所有运行固定使用完整 PIRS 奖励（`paper/specs/step_time.toml`）。候选上限取停车点步进延时 `step_delay_s = 2 s`。

```bash
# 训练（自动跳过已完成且可复用的运行；矩阵定义见 paper/specs/step_time.toml）
uv run python -m paper.experiments step_time run

# 汇总数值与表格（本实验不出图）
uv run python -m paper.experiments step_time summarize \
  --output output/paper_experiment/01_step_time/latest
```

论文输出（写入 `--output` 目录）：

- `step_time_table.md`：每个控制周期两行，分别为“Best”（训练期 `best/` 检查点）与“Final”（训练结束时的策略，即运行目录顶层的确定性评估），列为严格可行率（百分比与分子/分母）、绝对停站误差、有符号时间误差 Δt、轨迹能耗（kWh）、舒适度 TAV。严格可行率按全部种子计算；其余指标为到站策略的均值 ± 样本标准差（无到站策略时为“—”）。
- `step_time_summary.json`：各控制周期 `best`/`final` 两组的逐种子原始指标、严格可行数与严格可行率、到站数，训练后期（最后 25% 预算）可行评估比例、best→final 能耗漂移，以及最终选定控制周期（`recommended_step_time_s`）与选择过程（`reference_energy_kwh`、`energy_margin_kwh`、`equivalent`、`excluded`）。选择规则（`select_step_time`）分三层：① 可靠性门槛——所有种子的 best 与 final 策略均严格可行，且后期可行评估比例均值 ≥ 0.9；② 能耗——final 策略平均能耗与最低者之差不超过各候选 final 能耗跨种子标准差的合并值（均方根）即视为等价（不用 best：best 检查点是可行评估中能耗最低者，倾向于利用时分容限晚到，其能耗混入了时间误差）；③ 收敛稳定性——等价者中取 final 策略能耗跨种子标准差最小者，再比较平均能耗漂移，最后取较小控制周期。

训练运行写入 `paper/specs/step_time.toml` 的 `output_root`（`output/paper_experiment/01_step_time/`），每个矩阵单元一个 `<name>__<variant>__seed<seed:04d>__<NN>` 子目录（例如 `step_time__1p0__seed0011__01`）。方法消融（第 6 节）使用本节 `step_time_summary.json` 中的 `recommended_step_time_s` 作为其控制周期，查看方式：

```bash
uv run python -c "import json; print(json.load(open('output/paper_experiment/01_step_time/latest/step_time_summary.json'))['recommended_step_time_s'])"
```

该值为 `null`（上面的命令会打印 Python 的 `None`）表示四组控制周期本轮均无严格可行轨迹，不能确定推荐值；此时不应据此运行方法消融。

方法消融的 `paper/specs/method_ablation.toml` 中 `[train].step_time_s` 取步长消融的推荐值（当前为 `1.0 s`）；若重新运行步长消融得到不同的推荐值，先把该字段改为推荐值，再运行方法消融（第 6 节）。

## 6. 方法消融（论文实验二）

四种方法、5 个随机种子和相同环境转移预算（`paper/specs/method_ablation.toml`）。

```bash
uv run python -m paper.experiments method_ablation run

uv run python -m paper.experiments method_ablation summarize \
  --output output/paper_experiment/02_method_ablation/latest

uv run python -m paper.experiments method_ablation figures \
  --output output/paper_experiment/02_method_ablation/latest
```

论文输出（写入 `--output` 目录）：

- `method_training_curves.pdf`：1×2 子图：(a) 训练过程近 12 rollouts（98,304 transitions）每万步速度违规次数；(b) 训练到达率。
- `method_representative_profiles.pdf`：四个变体代表策略的速度—位置曲线叠加（背景为防护曲线与危险域）：(a) 全程；(b) 出发后 6 km；(c) 进站前 4 km。每个变体的代表策略按与第 7 节相同的规则选取：各种子 final 策略中严格可行且能耗最低者，无可行者按 `selection_key` 回退。
- `method_training_table.md`：四个训练时期（1–100、101–200、201–300、301–400 rollouts）各方法的违规率与到达率均值 ± 样本标准差。
- `method_performance_table.md`：每个方法分 best（`best/` 检查点）与 final（训练结束时的策略）两行，列出严格可行率（全部种子）以及到站策略的停站误差、有符号时间误差 Δt、总能耗（kWh）、舒适度 TAV；训练后期可行评估比例（最后四分之一训练预算内周期评估严格可行的比例）与首次可行评估进度（首次出现严格可行评估时已用训练预算的比例，只统计出现过的种子）属于训练过程，只列在 best 行。均为均值 ± 样本标准差（无到站时为“—”）。`summary.json` 的 `performance` 按 `best`/`final` 分组，`raw_seed_metrics_final` 为 final 策略的逐种子指标；配对差值基于 best；代表策略基于 final（见第 2 节）。
- `summary.json`：原始种子指标、各变体代表策略（`representative_policies`，其中 `ppo_pirs` 另存为 `representative_policy`）等派生数据（不含表格文本）。

训练运行写入 `paper/specs/method_ablation.toml` 的 `output_root`（`output/paper_experiment/02_method_ablation/`）。第 7 节从本节 `summarize` 写出的 `summary.json` 中取代表性 PPO+PIRS 运行目录（控制台运行 `run` 时也会打印每个运行目录，但代表性运行的选定要看 `summary.json` 的 `representative_policy`）。

## 7. DP、PIRS 与实际运行对比（论文实验三）

本节不经过 `paper/experiments` 的复用层，每次运行需要人工指定未使用过的输出目录。建议约定：`output/paper_experiment/03_dp_actual_comparison/<批次号>/`。

```bash
DP_DIR=output/paper_experiment/03_dp_actual_comparison/latest/dp

# 1) 生成 DP 参考轨迹（口径与 paper/specs/dp.toml 一致：465 s、0.1 m/s、等间距 30 m）
uv run mtto dp \
  --scenario paper/specs/scenario.toml --line-dir paper/data/line \
  --tasks paper/specs/tasks.toml --task longyang_to_airport \
  --config paper/specs/dp.toml \
  --output-dir "$DP_DIR"

# 2) 从方法消融汇总（第 6 节 summarize 产物）取代表性 PPO+PIRS 运行目录：
#    representative_policy.run_dir 是代表 final 策略所在的运行目录。
RL_MODEL_DIR=$(uv run python -c "
import json
data = json.load(open('output/paper_experiment/02_method_ablation/latest/summary.json'))
print(data['representative_policy']['run_dir'])
")
echo "代表运行目录: $RL_MODEL_DIR"

# 3) 三方对比图与对比表；实测运营曲线由脚本直接调用
#    paper.real_operation.real_operation_profile 现场计算，不需要预先转换文件
uv run python -m paper.figures.speed_profile_comparison \
  --dp-run "$DP_DIR" \
  --rl-run "$RL_MODEL_DIR" \
  --output-dir output/paper_experiment/03_dp_actual_comparison/latest \
  --no-show
```

`mtto dp` 默认不启用磁盘缓存（状态转移图重新计算），如需要跨运行复用转移图缓存，显式传入 `--cache-dir <目录>`。

`uv run python -m paper.real_operation [--output-file ...]` 是独立工具：将实测运营 Excel 数据（`paper/data/operation/`）线性重标定到指定起终点并另存为 NPZ 数组，供外部检查或后续分析使用；`speed_profile_comparison` 与 `paper.figures.real_operation_data` 都不读取这个 NPZ，而是直接调用 `paper.real_operation.real_operation_profile` 现场重新计算，因此运行三方对比图前不需要先执行本命令。

论文输出：

- `dp_rl_actual_comparison.pdf`：速度—位置、加速度—位置和累计能耗—位置三个子图；能耗纵轴单位为 kWh。
- `dp_rl_actual_comparison_table.md`：对比表，包含来源、停站误差、时间误差、总能耗（kWh）、舒适度 TAV（实际曲线的 TAV 因加速度估算口径不同显示为“—”），未满足停站/准点容限的能耗以 `^a` 标注。

`paper.figures.speed_profile_comparison` 还支持 `--baseline-rl LABEL=DIR`（可重复，最多 3 个）叠加额外 RL 基线曲线；`paper.figures.dp_redundancy_error` 与 `paper.figures.sps_compliance --analysis-mode compare` 可对同一 DP/RL 运行目录做冗余运行时间误差分析与停车点步进（SPS）合规性分析。

## 8. 途中计划时分变更（论文实验四）

`paper/specs/schedule_change.toml` 定义本实验：

- **RL**：从方法消融（`source_spec`）中按第 6 节的代表策略规则选出 PPO+PIRS 的代表 final 策略（变体固定，spec 中不再配置）（与第 7 节为同一策略，不做跨工况挑选），对 `delta_times_s` 中的每种变化（默认 `[0.0, 30.0, -30.0]`）在绝对线路位置 `change_distance_m`（默认 `8000.0` m）处改变计划时分并做确定性评估。评估经过第 4 节的复用层。
- **DP**：读取第 7 节的标称 DP 运行（`dp_run`），取其轨迹上第一个不早于变化位置的网格节点的位置、速度和已用时间，以“新计划时分 − 已用时间”为剩余计划时分、用同一 DP 配置（取自该运行的 `run.json`）从该节点重新求解至终点，与变化前的标称段拼接成全程轨迹并评估。
- **重新计算耗时**：RL 为从变化位置起逐步构建观测、策略推理并在仿真器中推进至停站的挂钟时间，另记单步推理均值；DP 为从变化节点重新求解的挂钟时间（含转移图构建与 λ 二分），另记同一求解器实例复用内存中转移图时只做 λ 二分的耗时。每项重复 `timing_repeats` 次（默认 3），汇总取中位数；DP 预计算模式与进程数、PyTorch 线程数一并写入表注。策略加载与环境构建不计入耗时。

运行本节前，方法消融（第 6 节）必须已产生完整运行，且第 7 节的 DP 运行已位于 `dp_run`。

```bash
uv run python -m paper.experiments schedule_change run

uv run python -m paper.experiments schedule_change summarize \
  --output output/paper_experiment/04_schedule_time_change/latest

uv run python -m paper.experiments schedule_change figures \
  --output output/paper_experiment/04_schedule_time_change/latest
```

论文输出（写入 `--output` 目录）：

- `schedule_time_change_comparison.pdf`：各工况的速度—位置曲线，颜色区分工况，PPO-PIRS 实线、DP 虚线，竖线标出计划变化位置。
- `schedule_time_change_table.md`：工况 × 方法，列为是否满足停站/准点容限、相对新计划的有符号 Δt、停站误差、全程能耗（kWh）、舒适度 TAV、重新计算耗时。
- `schedule_time_change_summary.json`：逐工况、逐方法的指标与耗时、代表策略与变化节点信息。

产物位置：RL 评估写入 `output_root` 下的 `schedule_change__<variant>__<工况>__NN/`（评估的 `reuse_key` 额外绑定被评估策略 `policy.zip` 的 SHA-256）；DP 重新求解与耗时写入 `output_root/replan__<键>/`（`dp__<工况>/` 为拼接后的 DP 运行，`timing.json` 为耗时与溯源信息，最后写入，作为完成标记）。`replan` 的键绑定场景哈希、策略 SHA-256、标称 DP 的 `run_id`、工况、变化位置、重复次数与 `mtto` 版本；键相同且 `timing.json` 存在时复用，键变化时写入新目录、不删除旧目录。`figures` 同样拒绝 `timing.json` 中 `dirty=true` 的结果。

## 9. 输出位置一览

| 实验 | 训练/评估产物 | 汇总与图表 |
| --- | --- | --- |
| 控制周期步长消融 | `paper/specs/step_time.toml` 的 `output_root`（`output/paper_experiment/01_step_time/`） | `step_time summarize --output <目录>` |
| 方法消融 | `paper/specs/method_ablation.toml` 的 `output_root`（`output/paper_experiment/02_method_ablation/`） | `method_ablation {summarize,figures} --output <目录>` |
| DP/PIRS/实际对比 | 由 `mtto dp --output-dir` 手工指定（实测曲线现场计算，无需单独产物） | `paper.figures.speed_profile_comparison --output-dir <目录>` |
| 途中计划时分变更 | `paper/specs/schedule_change.toml` 的 `output_root`（`output/paper_experiment/04_schedule_time_change/`，含 `replan__<键>/`） | `schedule_change {summarize,figures} --output <目录>` |
| 环境、势函数、评分函数说明图 | 不适用（不读运行产物） | `paper.figures.{env_data,potential_function,score_function} --output-dir <目录>` |

## 10. 测试与验收

```bash
uv run pytest                             # 默认跳过标记为 slow、golden 的用例
uv run pytest -m golden                   # 与 output/golden/ 中录制的回归快照比对
uv run ruff check src tests paper
uv run ruff format --check src tests paper
```

`tests/golden/` 存放用例定义与冻结的固定动作序列（`actions/*.npy`），回归快照（DP 最优解、固定动作序列下的 RL 轨迹与逐步奖励、计划变更情形等）由 `uv run python -m tests.golden.record` 录制到 `output/golden/`，详见项目根 `README.md` 的“Golden 回归快照”；有意改变计算逻辑时须在提交说明中写明原因、附差异摘要，递增 `pyproject.toml` 中的 `project.version` 并重新录制。
