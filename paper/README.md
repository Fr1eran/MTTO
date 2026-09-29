# 高速磁浮速度曲线优化：论文仿真实验指南

本手册描述论文全套仿真实验：空间步长消融、方法消融、DP/PIRS/实际运行三方对比、计划时间变化鲁棒性。核心强化学习方法称为物理先验奖励塑形（Physics-Informed Reward Shaping, PIRS），内部标识为 `ppo_pirs`。所有命令均从项目根目录执行，统一通过 `uv run` 使用项目虚拟环境；论文相关依赖（matplotlib、pandas、openpyxl）随 `dev` 依赖组一并安装（见 `uv sync`）。

实验编排代码位于 `paper/experiments/`（`spec.py` 解析 `paper/specs/*.toml` 并展开实验矩阵，`runner.py` 负责调用 `mtto.workflows` 并处理中断恢复与结果复用），命令行入口为 `python -m paper.experiments`。出图脚本位于 `paper/figures/`。第 4 节描述结果复用规则；不熟悉该机制时，重新运行下列命令是安全的——已完成且未过期的结果会被自动复用而不是重新计算。

## 1. 实验结构与固定口径

完整仿真实验分为四部分：

1. 空间步长消融（第 5 节）：确定后续实验使用的控制步长。
2. 方法消融（第 6 节）：对比 PPO、PPO+Safety、PPO+Punctuality 和 PPO+PIRS。
3. DP、PIRS 与实际运行结果对比（第 7 节）。
4. 计划时间变化鲁棒性验证（第 8 节）。

| 项目 | 固定值 | 定义位置 |
| --- | --- | --- |
| 场景 | 上海高速磁浮示范线下行场景（龙阳路 → 浦东国际机场） | `paper/specs/scenario.toml`、`paper/specs/tasks.toml` 的 `longyang_to_airport` |
| 名义计划运行时间 | `465 s` | `paper/specs/tasks.toml` |
| 步长候选 | `10, 30, 50, 100 m` | `paper/specs/step_distance.toml` 的 `[[variants]]` |
| 随机种子 | `11, 131, 239, 359, 443` | `paper/specs/method_ablation.toml`、`step_distance.toml` 的 `seeds` |
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

（代码中还定义了第五个预设 `li2023_scaled`，用于叠加 Li et al. (2023) 基线曲线，见 `paper/figures/speed_profile_comparison.py` 的 `--baseline-rl`；论文四组核心方法消融不使用它。）

势函数塑形统一按 `gamma*Phi(next) - Phi(previous)` 计算，终止步继续使用观测到的下一状态势能（不显式归零）：

- **安全势**：按制动（牵引）储备步数的二次合页，`Phi = -c_up*(1 - clip(k_up,0,H)/H)² - c_low*(1 - clip(k_low,0,H)/H)²`，`H=3`（`SAFETY_RESERVE_HORIZON_STEPS`）、`c_up=3`、`c_low=1`（`SAFETY_RESERVE_UPPER_SCALE`/`SAFETY_RESERVE_LOWER_SCALE`）。制动储备 `k_up = min((v_max² - v²)/(2*b*dx), (v_max'² - v²)/(2*b*dx) + 1)`，其中 `v_max'` 为前方一步 `x+dx` 处的上限、`b` 为最大制动减速度，停车时为 +inf；牵引储备 `k_low` 同理，只对大于 0 的下限计入。`k <= 0` 表示已越界或下一步无论如何都将越界；储备 `>= H` 时势能为 0（`mtto.rl.rewards.braking_reserve_steps`/`traction_reserve_steps`/`safety_potential`）。
- **准点势**：`Phi = -K * e² / (e² + sigma²)`，`K=5`（`PUNCTUALITY_POTENTIAL_SCALE`）、`sigma=20 s`（`PUNCTUALITY_POTENTIAL_SIGMA_S`）；`e` 为实际剩余裕度与参考裕度之差，参考裕度按沿运行方向裁剪到 `[0,1]` 的剩余距离比例 `q` 线性插值：`b_ref = b0 * q`，`b0` 为全程静止起点的计划时间减最短运行时间。计划时间变化后以新计划更新全程参考线，不从变化点重锚。
- **终止步奖励**（仅 `STOPPED_IN_ZONE` 终止时触发）：停站分 `stopping_score(e_x) = 1 / (1 + (max(0, |e_x| - max_stop_error_m) / 0.3)²)`；准点分 `punctuality_score(e_t) = exp(-|e_t| / 45)`；终端奖励 `terminal_stopping = 15 * stopping_score`，`terminal_punctuality = 5 * punctuality_score + 20 * stopping_score² * punctuality_score`（`mtto.rl.rewards.RewardCalculator`）。

停站判定（`mtto.domain.scenario.Task.stop_state`）：停车（`|v| <= 0.01 m/s`）且 `|x - x_target| <= 30*max_stop_error_m` 为 `STOPPED_IN_ZONE`；列车带速到达目标点时不终止，按完整步长继续越过目标点，越过距离超过 `30*max_stop_error_m` 即为 `OVERRAN`；其余停车为 `STOPPED_SHORT`。到达目标点的那一步仍截短到恰好落在目标点上。

智能体观测为 11 维（`mtto.rl.observation.ObservationBuilder`，`POLICY_IO_VERSION=2`）：全程进度、带符号对数距离（尺度 0.1 m）、速度、上一步加速度、速度上/下限、准点比值 `e/sqrt(e²+sigma²)`（准点势恰为 `-K*o6²`）、恰好停在目标点所需的加速度、制动储备 `clip(k_up,0,H)/H`、冗余时间对数编码（尺度 10 s）、全力制动下的预计停站误差 `v²/(2b) - d` 的对数编码（尺度 0.1 m）。速度按线路最高限速归一化。

安全越界判定只在 `mtto.domain.safeguard`（动态双限速度防护，按节点）；算法的运行上限（SRTSP、静态允许速度域）属于算法约束，不进入质量判定。DP 使用实际任务起点至实际目标点、首尾速度均为零的最短运行时间曲线作为可达速度上界。所有方法的环境内部终止和违规截断均视为真实任务结束：外部时间上限截断映射为 `truncated=True`，其余终止原因映射为 `terminated`，失败终点不参与 PPO 的 bootstrap。

“最好结果”按以下口径执行（`mtto.evaluation.quality.selection_key`/`best_update_reason`，由 `ScheduledPolicyEvaluationCallback` 在训练期周期评估中维护）：

- 每个随机种子使用训练期 `best/` 检查点，不用 `final/`（训练结束时的最终策略）替代消融性能结果。
- 择优优先级：严格可行（安全、成功、精确停站且准点）轨迹优先于所有不可行轨迹；严格可行轨迹之间优先选择总能耗更低者；尚无严格可行轨迹时，依次按“安全且成功”、停站精度、准点性、时间误差绝对值和能耗回退。
- 消融学习曲线使用全部 5 个随机种子：先对单种子做 5 点尾随滑动平均，再计算多种子均值与样本标准差带（ddof=1）。
- 消融性能表对 5 个种子的 `best/` 评估指标报告均值 ± 样本标准差（ddof=1），包含全部 5 个种子，不挑选单个种子的最优数值。
- 三方轨迹对比使用方法消融选定的代表性 PPO+PIRS `best/` 策略。

## 3. 命令入口一览

| 用途 | 命令 |
| --- | --- |
| 单次 RL 训练/评估/DP/训练分析 | `uv run mtto {train,evaluate,dp,analyze-training} ...`（见项目根 `README.md`） |
| 步长/方法消融（训练矩阵） | `uv run python -m paper.experiments {step_distance,method_ablation} {run,summarize,figures} [--spec ...] [--output ...]` |
| 计划时间变化鲁棒性（评估矩阵） | `uv run python -m paper.experiments schedule_change {run,summarize,figures} [--spec ...] [--output ...]` |
| 实测运营数据重标定 | `uv run python -m paper.real_operation [--output-file ...]` |
| 环境、势函数、评分函数说明图 | `uv run python -m paper.figures.{env_data,potential_function,score_function}` |
| DP/RL 单条轨迹可视化 | `uv run python -m paper.figures.{dp_result,rl_result} --dp-run/--rl-run <目录>` |
| 三方（DP/PIRS/实际）对比与 DP 冗余时间误差、SPS 合规分析 | `uv run python -m paper.figures.{speed_profile_comparison,dp_redundancy_error,sps_compliance}` |
| 最短运行时间曲线、实测数据曲线（无参数，交互式脚本） | `uv run python -m paper.figures.{min_operation_time_curve,real_operation_data}` |

所有出图脚本支持 `--no-show`（保存不弹窗）与大多数支持 `--output-dir`（固定文件名，见各脚本 `--help`）；`min_operation_time_curve`（交互式最短运行时间曲线计算器，键盘输入起点重新计算）与 `real_operation_data`（打印实测曲线统计并展示三张图）没有命令行参数，始终调用 `plt.show()`；在无显示环境下用 `MPLBACKEND=Agg` 运行会跳过弹窗但仍完整执行其余逻辑。

`run`/`summarize`/`figures` 的 `--spec` 默认分别为 `paper/specs/step_distance.toml`、`paper/specs/method_ablation.toml`、`paper/specs/schedule_change.toml`；`run` 不需要 `--output`（训练/评估产物写入 spec 的 `output_root`），`summarize`/`figures` 需要 `--output <目录>`（汇总 JSON、Markdown 表格、PDF 图表写入该目录，可以是任意路径，不必与 `output_root` 相同）。

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

## 5. 空间步长消融（论文实验一）

比较 `10/30/50/100 m × 5 seeds`，所有运行固定使用完整 PIRS 奖励（`paper/specs/step_distance.toml`）。

```bash
# 训练（自动跳过已完成且可复用的运行；矩阵定义见 paper/specs/step_distance.toml）
uv run python -m paper.experiments step_distance run

# 汇总数值与表格（不检查 dirty）
uv run python -m paper.experiments step_distance summarize \
  --output output/paper_experiment/01_step_distance/latest

# 出图（要求全部运行 dirty=false，或当前工作区处于 dirty 状态）
uv run python -m paper.experiments step_distance figures \
  --output output/paper_experiment/01_step_distance/latest
```

论文输出（写入 `--output` 目录）：

- `step_distance_learning_curves.pdf`：1×2 子图：(a) 行程完成率；(b) 周期独立评估可行率（feasible rate）。
- `step_distance_table.md`：步长、严格可行率（百分比与分子/分母）、绝对停站误差、绝对时间误差、轨迹能耗（kWh）、舒适度 TAV，5 个种子的均值 ± 样本标准差。
- `step_distance_summary.json`：各步长各种子的原始指标、严格可行数与严格可行率，以及最终选定步长（`recommended_step_distance`，选择规则：严格可行数最多 → 平均里程完成率最高 → 可行轨迹平均能耗最低 → 可行轨迹平均舒适度最低 → 较小步长）与比较键。

训练运行写入 `paper/specs/step_distance.toml` 的 `output_root`（`output/paper_experiment/01_step_distance/`），每个矩阵单元一个 `<name>__<variant>__seed<seed:04d>__<NN>` 子目录（例如 `step_distance__30p0__seed0011__01`）。方法消融（第 6 节）使用本节 `step_distance_summary.json` 中的 `recommended_step_distance` 作为其空间步长，查看方式：

```bash
uv run python -c "import json; print(json.load(open('output/paper_experiment/01_step_distance/latest/step_distance_summary.json'))['recommended_step_distance'])"
```

该值为 `null`（上面的命令会打印 Python 的 `None`）表示四组步长本轮均无严格可行轨迹，不能确定推荐步长；此时不应据此运行方法消融。

方法消融的 `paper/specs/method_ablation.toml` 中 `[train].step_distance_m` 取步长消融的推荐步长（论文正式实验为 `30 m`）；若重新运行步长消融得到不同的推荐值，先把该字段改为推荐值，再运行方法消融（第 6 节）。

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
- `method_trajectory_metrics.pdf`：2×2 子图：绝对停站误差、绝对时间误差、轨迹总能耗（kWh）和舒适度 TAV，含 rollout 300–396 局部放大图与阈值虚线（停站 0.3 m，准点 10 s）。
- `method_training_table.md`：四个训练时期（1–100、101–200、201–300、301–400 rollouts）各方法的违规率与到达率均值 ± 样本标准差。
- `method_performance_table.md`：各方法 5 种子 `best/` 轨迹的严格可行率、停站误差、时间误差、总能耗（kWh）、舒适度 TAV 均值 ± 样本标准差。
- `summary.json`：原始种子指标、代表策略信息等派生数据（不含表格文本）。

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
#    representative_policy.model_path 是策略文件路径（.../<运行目录>/best/policy.zip
#    或没有 best/ 时的 .../<运行目录>/policy.zip），运行目录是它的上一级（best/ 存在时再上一级）。
RL_MODEL_DIR=$(uv run python -c "
import json
from pathlib import Path
data = json.load(open('output/paper_experiment/02_method_ablation/latest/summary.json'))
model_path = Path(data['representative_policy']['model_path'])
run_dir = model_path.parent.parent if model_path.parent.name == 'best' else model_path.parent
print(run_dir)
")
echo "代表运行目录: $RL_MODEL_DIR"

# 3) 三方对比图与对比表；实测运营曲线由脚本直接调用
#    paper.real_operation.real_operation_profile 现场计算，不需要预先转换文件
uv run python -m paper.figures.speed_profile_comparison \
  --dp-run "$DP_DIR" \
  --rl-run "$RL_MODEL_DIR" --rl-best \
  --output-dir output/paper_experiment/03_dp_actual_comparison/latest \
  --no-show
```

`mtto dp` 默认不启用磁盘缓存（状态转移图重新计算），如需要跨运行复用转移图缓存，显式传入 `--cache-dir <目录>`。

`uv run python -m paper.real_operation [--output-file ...]` 是独立工具：将实测运营 Excel 数据（`paper/data/operation/`）线性重标定到指定起终点并另存为 NPZ 数组，供外部检查或后续分析使用；`speed_profile_comparison` 与 `paper.figures.real_operation_data` 都不读取这个 NPZ，而是直接调用 `paper.real_operation.real_operation_profile` 现场重新计算，因此运行三方对比图前不需要先执行本命令。

论文输出：

- `dp_rl_actual_comparison.pdf`：速度—位置、加速度—位置和累计能耗—位置三个子图；能耗纵轴单位为 kWh。
- `dp_rl_actual_comparison_table.md`：对比表，包含来源、停站误差、时间误差、总能耗（kWh）、舒适度 TAV（实际曲线的 TAV 因加速度估算口径不同显示为“—”），未满足停站/准点容限的能耗以 `^a` 标注。

`paper.figures.speed_profile_comparison` 还支持 `--baseline-rl LABEL=DIR`（可重复，最多 3 个）叠加额外 RL 基线曲线；`paper.figures.dp_redundancy_error` 与 `paper.figures.sps_compliance --analysis-mode compare` 可对同一 DP/RL 运行目录做冗余运行时间误差分析与停车点步进（SPS）合规性分析。

## 8. 计划时间变化鲁棒性（论文实验四）

`paper/specs/schedule_change.toml` 从方法消融（`source_spec`）中读取 `source_variant`（`ppo_pirs`）的 5 个种子的 `best/` 和 `final/`，共 10 个候选策略；每个候选运行 `delta_times_s` 中列出的每种计划时间变化（默认 `[0.0, 30.0, -30.0]`，即 `Original`、`Plus 30s`、`Minus 30s`），在线路位置 `change_distance_m`（绝对位置，默认 `8000.0` m）处触发变化。运行本节前，方法消融（第 6 节）必须已产生完整的 20 组运行。

```bash
uv run python -m paper.experiments schedule_change run

uv run python -m paper.experiments schedule_change summarize \
  --output output/paper_experiment/04_schedule_time_change/latest

uv run python -m paper.experiments schedule_change figures \
  --output output/paper_experiment/04_schedule_time_change/latest
```

论文输出（写入 `--output` 目录）：

- `schedule_time_change_comparison.pdf`：各计划时间下的速度—位置曲线，标记计划变化位置，仅绘制根据跨工况严格可行性、安全、完成、停站/准点误差和能耗选出的最稳健候选。
- `schedule_time_change_table.md`：计划变化、最终时间误差 (s)、停站误差 (m)、轨迹能耗 (kWh)、舒适度 TAV (m/s²)。
- `schedule_time_change_summary.json`：逐候选、逐工况原始指标（能耗为 `total_energy_j`，展示时按 `/ 3,600,000` 换算为 kWh）与最终排名。

评估运行写入 `paper/specs/schedule_change.toml` 的 `output_root`（`output/paper_experiment/04_schedule_time_change/`），同样经过第 4 节的复用与 `dirty` 规则（评估的 `reuse_key` 额外绑定被评估策略 `policy.zip` 的 SHA-256）。

## 9. 输出位置一览

| 实验 | 训练/评估产物 | 汇总与图表 |
| --- | --- | --- |
| 空间步长消融 | `paper/specs/step_distance.toml` 的 `output_root`（`output/paper_experiment/01_step_distance/`） | `step_distance {summarize,figures} --output <目录>` |
| 方法消融 | `paper/specs/method_ablation.toml` 的 `output_root`（`output/paper_experiment/02_method_ablation/`） | `method_ablation {summarize,figures} --output <目录>` |
| DP/PIRS/实际对比 | 由 `mtto dp --output-dir` 手工指定（实测曲线现场计算，无需单独产物） | `paper.figures.speed_profile_comparison --output-dir <目录>` |
| 计划时间变化鲁棒性 | `paper/specs/schedule_change.toml` 的 `output_root`（`output/paper_experiment/04_schedule_time_change/`） | `schedule_change {summarize,figures} --output <目录>` |
| 环境、势函数、评分函数说明图 | 不适用（不读运行产物） | `paper.figures.{env_data,potential_function,score_function} --output-dir <目录>` |

## 10. 测试与验收

```bash
uv run pytest                             # 默认跳过标记为 slow 的用例
uv run pytest -m slow                     # 含 DP golden 等耗时用例
uv run ruff check src tests paper
uv run ruff format --check src tests paper
```

`tests/golden/` 存放冻结的数值快照（DP 最优解、固定动作序列下的 RL 轨迹与逐步奖励、计划变更情形等），不得随意改动其中的数据；更新 golden 快照须在提交说明中写明原因、附差异摘要，并递增 `pyproject.toml` 中的 `project.version`。
