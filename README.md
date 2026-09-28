# MTTO

中高速磁浮列车运行速度曲线优化 —— 动态规划（基线）与强化学习（主，物理先验奖励塑形 PIRS）双链路。

---

## 目录

- [项目结构](#项目结构)
- [架构设计](#架构设计)
  - [分层与依赖方向](#分层与依赖方向)
  - [关键设计决定](#关键设计决定)
  - [正确性的保证方式](#正确性的保证方式)
  - [扩展接缝](#扩展接缝)
  - [已知限制与推迟事项](#已知限制与推迟事项)
- [安装](#安装)
- [命令行：`mtto`](#命令行mtto)
  - [`mtto train`](#mtto-train)
  - [`mtto evaluate`](#mtto-evaluate)
  - [`mtto dp`](#mtto-dp)
  - [`mtto analyze-training`](#mtto-analyze-training)
- [产物约定](#产物约定)
- [论文实验与图表](#论文实验与图表)
- [测试与代码检查](#测试与代码检查)
- [Golden 回归快照](#golden-回归快照)
- [扩展占位](#扩展占位)

---

## 项目结构

```text
src/mtto/                 可安装的库（不依赖 matplotlib、pandas、openpyxl）
  domain/                 高速磁浮列车运行机理：线路、动力学、能耗、动力学积分、安全防护、
                          最短运行时间速度曲线（SRTSP）、运行场景 Scenario/Task、SpeedProfile
  evaluation/             速度曲线质量：指标、安全防护审计、可行判定、择优规则
  rl/                     强化学习：环境、观测/动作编解码、奖励、PPO 构建、评估、训练期诊断、
                          训练日志分析（training_analysis/）；morl/（占位，见下）
  dp/                     动态规划：状态图、求解器、状态图磁盘缓存
  io/                     产物读写约定（run.json 等，见下）、场景/任务加载、TensorBoard 读取
  workflows/              cli、paper、app 共用的入口：train、evaluate、dp、analysis
  cli.py                  命令行入口（`mtto` 命令）
paper/                    论文：验证方法可行性的仿真实验（依赖组 paper：matplotlib、pandas、openpyxl）
  README.md                 完整仿真实验指南
  data/line/  data/operation/   上海磁浮示范线线路数据与实测运营数据
  real_operation.py         实测运营数据读取、位置对齐与时间恢复
  specs/                    场景、任务与实验定义（TOML）
  experiments/               实验编排：TOML 解析、矩阵展开、中断恢复与结果复用
  analysis.py               跨种子聚合、约束统计等论文分析
  plotting/                 matplotlib 绘图代码
  figures/                  论文图表脚本
app/                      面向非专业用户的应用层（本次只占位，见 `app/ui/README.md`、`app/api/README.md`）
tests/                    单元测试、golden 快照（`tests/golden/`）、论文层测试（`tests/paper/`）
```

## 架构设计

代码组织围绕三个目标：

1. **DP 与 RL 共用同一套运行机理**，两条链路的差异只在算法本身；
2. **任何来源的速度曲线（RL、DP、实测数据）用同一把尺子评价**；
3. **计算结果可以溯源，论文实验可以安全地中断、恢复与复用**。

由此得到下面的分层、若干条贯穿全库的约定，以及以 golden 快照为核心的验证方式。

### 分层与依赖方向

```text
cli        → workflows
app        → workflows
paper      → mtto 的任意模块（研究代码，可直接使用 domain、rl.rewards 等）
workflows  → rl, dp, evaluation, io
rl         → evaluation, domain
dp         → domain
io         → rl（终止原因、诊断数据、策略 IO 版本的读写）, evaluation（质量报告的读写）, domain
evaluation → domain
domain     → numpy、numba、标准库
```

| 层 | 职责 | 刻意不做的事 |
| --- | --- | --- |
| `domain` | 高速磁浮列车运行机理：线路查询、动力学、匀变速运动学、能耗、安全防护（静态允许速度域、停车点步进与动态双限）、最短运行时间速度曲线（SRTSP）、`Scenario`/`Task`、`SpeedProfile`；热点函数为 numba kernel | 不含算法、不做 I/O；只有 `scenario.py` 聚合各组成部分，其余模块不导入它，从结构上避免循环导入 |
| `evaluation` | 速度曲线质量：指标、按节点的动态双限审计、可行判定、择优规则 | 与算法、奖励无关 |
| `rl`、`dp` | 两种算法：RL 的环境状态转移、观测与动作编解码、奖励、PPO 构建与回调、训练诊断；DP 的状态图、求解与状态图缓存 | 彼此不导入，也不导入 `io`；各自实现状态转移，只调用 `domain` 的共用函数 |
| `io` | 产物与输入的唯一读写点：`run.json` 等固定文件名与严格读写、按路径加载场景与任务、TensorBoard 读取 | 不导入 `workflows` |
| `workflows` | `cli`、`paper`、`app` 共用的入口：`train`、`evaluate`、`dp`、`analysis`；接收配置对象、返回结果对象、以回调报告进度 | 不解析命令行参数、不打印、不绘图 |
| `cli` | 解析参数、调用 `workflows`、显示结果 | 不含业务逻辑 |
| `paper` | 论文专用：数据、TOML 定义、实验编排与结果复用、跨种子分析、全部绘图 | — |

`src/mtto` 不导入 `paper` 与 `app`，也不依赖 matplotlib、pandas、openpyxl（它们在 `paper` 依赖组中）。上表中的每条依赖规则都由 `tests/test_dependency_boundaries.py` 用 AST 扫描检查，破坏分层会直接导致测试失败。

### 关键设计决定

**共用运行机理，不设公共仿真层。** RL 的一步是 `kinematics.run_distance` 加 `energy.segment_energy`（能耗使用指令加速度）；DP 的一条边是 `kinematics.accel_between` 加加速度上下限检查与 `energy.segment_energy`。两者调用同一组函数，一致性由 `tests/test_kinematics.py` 检查。环境的状态转移仍由各算法自己实现，而不是抽出一个公共模拟器：这样将来加入按时间步离散的环境（`kinematics.run_time` 已提供）或 MORL 算法时，只需复用 `domain` 函数，不必迁就一个为现有算法设计的仿真接口。

**速度曲线只有一种表示，质量评估只有一条路径。** RL 轨迹、DP 最优解与实测运营曲线都表示为 `SpeedProfile`（节点上的位置、速度、时间、牵引与悬浮累计能耗，以及段上的 Δv/Δt 加速度），统一交给 `evaluation.quality.assess`，得到 `QualityReport`：指标、动态双限审计、完成/精确停车/安全/准点/可行判定。择优排序（`selection_key`）也只基于它。论文中不同方法之间的对比因此建立在同一套口径上。

**算法约束与质量判定分开。** 算法可以采用更严格的运行上限（RL 与 DP 使用 SRTSP 上限，DP 还使用静态允许速度域），它们只影响算法内部；质量判定中的“安全”只按动态双限速度防护，并且逐节点审计、记录全部越界。越界判定集中在 `domain/safeguard`，“是否停在允许停车区内”只由 `Task.stop_state` 回答。

**终止语义显式化。** 回合结束时环境给出 `TerminationReason`（停区内停车、停车不足、越过终点、低于下限、高于动态双限上限、高于 SRTSP 上限，见 `rl/state.py`）。所有终止原因都映射为 Gym 的 `terminated`，失败终点不参与 PPO 的价值 bootstrap；`truncated` 只留给将来的时间上限。奖励分支、成功判定与训练诊断都直接按终止原因计算，不再从标志组合反推。

**不可变配置与显式状态。** `Scenario`、`Task`、`ScheduleChange` 都是冻结 dataclass；计划运行时间与计划变更进入环境状态，而不是原地修改共享对象；每个派生量只在一处计算（例如 SRTSP 查找表与奖励归一化常数由 `workflows` 计算一次，传给所有并行环境）。

**数值只有一个定义点。** 车辆参数、能耗参数、防护曲线的生成输入、任务起终点与阈值只写在 `paper/specs/*.toml` 中；库中参数 dataclass 的数值字段不设默认值（表示“无”的可选字段可以默认 `None`）。`io/scenario.py` 按调用方传入的路径读取线路数据，在内存中生成防护曲线（不依赖本地缓存文件），并计算 `scenario_hash` 供溯源与复用使用。

**库负责计算与溯源，论文层负责复用。** `io/artifacts.py` 规定每类运行的产物组成并做严格读写；`run.json` 在其余产物写完并校验后最后原子写入，是运行完成的唯一标志；RL 的 `result.json` 记录终止原因、策略来源与实际训练完成情况。结果能否复用由 `paper/experiments` 判断：复用键由配置、场景哈希、任务、`mtto` 版本号（评估运行另加被评估策略文件的 SHA-256）组成，`paper.json` 同时记录 git 提交与工作区是否 dirty。库本身不做任何复用判断。

**版本号代表计算逻辑。** 复用键不包含代码哈希，而是包含 `pyproject.toml` 中的版本号：修改物理、判定或奖励等计算逻辑时必须递增版本号，旧结果随之不再被复用；只修改 `paper/`、CLI、测试或文档时不递增。DP 状态图缓存（库中唯一的缓存）的键同样包含版本号。

**只在已有两个实现时才抽象**，并遵循 `AGENTS.md`：只在真正的边界（外部文件、用户输入）做校验，内部调用之间信任契约，不写防御性的重复检查。

### 正确性的保证方式

- **golden 回归快照**（`tests/golden/`，`uv run pytest -m golden`）：固定动作序列下的 RL 轨迹，覆盖全部终止结局、全部奖励预设的逐步奖励与观测、计划变更的三种情形、DP 最优解，以及 RL/DP 用例的 `evaluation.quality.assess` 结果。离散量（终止原因、步数、停车点序号、质量判定的布尔结果）必须完全相等；连续量纯数值计算 `rtol=1e-9`，奖励与能耗 `rtol=1e-7`。快照是当前版本的回归基线，不是与重构前实现的对比；详见[Golden 回归快照](#golden-回归快照)。
- **依赖边界测试**：见上一节。
- 固定随机种子时，CPU 上的训练逐位可复现。重构完成时评估链路与重构前逐位一致、训练过程除有意修复的终止语义（旧实现把领域失败当作截断）外逐位一致，此结论由当时的迁移基线验证；迁移基线已完成使命并随本次测试整理移除，不再是仓库内的自动化保证，仅留作历史记录。

### 扩展接缝

| 扩展 | 现有接缝 | 届时要做的事 |
| --- | --- | --- |
| MORL 与其他 Pareto 解集算法 | `Task.schedule_time_s` 可为 `None`（可行判定随之不含到站时间）；奖励按分量返回 `RewardBreakdown`；`evaluation/quality.py` 与奖励无关 | 在 `rl/morl/` 或新的算法包中实现状态转移与目标定义；基于质量指标的 Pareto 分析放入 `evaluation/` |
| 应用层 `app/`（UI、API） | 只调用 `workflows`；`workflows` 返回含 `SpeedProfile` 的结果对象、以回调报告进度；`Scenario`/`Task` 不可变；线路数据按路径读取 | 在 `workflows` 中增加取消机制，长任务放到独立进程，再决定服务框架与前端 |
| 数据库 | 每个运行的 `run.json` 含完整配置与输入 | 按真实查询需求设计表结构 |
| 按时间步离散的环境 | `kinematics.run_time` 已实现并测试 | 新环境调用 `run_time`；质量评估增加冲击率舒适度口径；终止原因增加时间上限并映射为 `truncated` |
| 完整反向运行 | `Task` 构造时对反向运行显式报错 | 引入路线坐标变换，在构造 `Scenario` 时镜像数据 |

### 已知限制与推迟事项

| 事项 | 触发条件 |
| --- | --- |
| RL 只在步末检查动态双限，步内越界可能漏检 | 质量审计显示漏检有实际影响时 |
| 能耗内核在牵引/制动切换附近不连续（加速度相差约 1e-13 时牵引能耗可相差约 0.09 kJ） | 需要能耗数值连续（如基于梯度的优化）或排查能耗异常时 |
| DP 状态图缓存使用 pickle，只应读取可信目录 | 缓存目录需要接收外部文件时 |
| 训练没有中途 checkpoint，中断的运行整体重跑 | 单次训练耗时长到无法接受重跑时 |
| 运营标准（停车误差、准点阈值等）与 `Task` 放在一起 | 需要多套运营标准时 |
| 反向运行、按时间步离散的环境、MORL、应用层 | 见上一节 |

## 安装

```bash
uv sync
```

`uv sync` 默认安装 `dev` 依赖组，其中包含 `paper` 依赖组（matplotlib、pandas、openpyxl），因此论文实验与出图脚本无需额外安装步骤。命令行入口 `mtto` 在 `uv sync` 后即可用（`[project.scripts]`，见 `pyproject.toml`）。

## 命令行：`mtto`

`mtto` 提供四个子命令，均通过 `--scenario`/`--line-dir`/`--tasks`/`--task`（除 `analyze-training` 外）指定场景与任务；论文口径的场景/任务定义位于 `paper/specs/scenario.toml`、`paper/specs/tasks.toml`（任务名 `longyang_to_airport`）。完整参数见 `uv run mtto <子命令> --help`。

### `mtto train`

训练 RL 策略。`--config` 指向一个 TOML 文件的 `[train]` 表，严格解析（多余键或缺少不允许为空的键都会报错；允许为 `None` 的键可以省略）；字段定义见 `mtto.workflows.train.TrainConfig`。`paper/specs/train.toml` 是一份可直接使用的示例配置（对应论文方法消融的 PIRS 变体、第一个种子）：

```bash
uv run mtto train \
  --scenario paper/specs/scenario.toml \
  --line-dir paper/data/line \
  --tasks paper/specs/tasks.toml \
  --task longyang_to_airport \
  --config paper/specs/train.toml \
  --output-dir output/train/example
```

`--output-dir` 必须是尚不存在的目录（已存在时报错，不做覆盖）；可选 `--tensorboard-dir`、`--run-id`、`--schedule-time`（覆盖任务的计划运行时间）。

### `mtto evaluate`

在单环境中确定性（或加 `--stochastic` 随机）评估一次已完成的训练运行（`--policy-run`，加 `--use-best` 使用其 `best/` 检查点而非最终策略）：

```bash
uv run mtto evaluate \
  --scenario paper/specs/scenario.toml \
  --line-dir paper/data/line \
  --tasks paper/specs/tasks.toml \
  --task longyang_to_airport \
  --policy-run output/train/example \
  --use-best \
  --output-dir output/eval/example
```

### `mtto dp`

用动态规划求解速度曲线。`--config` 指向 TOML 文件的 `[dp]` 表，同样严格解析；字段定义见 `mtto.workflows.dp.DPConfig`。`paper/specs/dp.toml` 是一份示例配置（沿用重构前 `reproduce_dp` 脚本的默认参数）：

```bash
uv run mtto dp \
  --scenario paper/specs/scenario.toml \
  --line-dir paper/data/line \
  --tasks paper/specs/tasks.toml \
  --task longyang_to_airport \
  --config paper/specs/dp.toml \
  --output-dir output/dp/example
```

可选 `--cache-dir` 指定状态转移图的磁盘缓存目录；不传时不做磁盘缓存，每次重新计算。

### `mtto analyze-training`

对一次已完成的训练运行生成派生分析报告（不写 `run.json`，因为它不是计算运行）；`--train-run` 指向 `mtto train` 的输出目录。`--config` 指向一个 TOML 文件的 `[analysis]` 表，严格解析，为必填项（多余键或缺少不允许为空的键都会报错；允许为 `None` 的键可以省略）；字段定义见 `mtto.rl.training_analysis.pipeline.AnalysisConfig`。`paper/specs/analysis.toml` 是一份可直接使用的示例配置（对应重构前的默认值）：

```bash
uv run mtto analyze-training \
  --train-run output/train/example \
  --output-root output/analysis/example \
  --config paper/specs/analysis.toml
```

## 产物约定

产物读写统一经由 `mtto.io.artifacts`，固定文件名与严格键检查见该模块。每个运行目录以 `run.json`（运行来源、配置、`mtto` 版本、创建时间，最后原子写入，作为完成标记）区分三种 `kind`：

| `kind` | 必需产物 | 可选产物 |
| --- | --- | --- |
| `rl_train` | `run.json`、`policy.zip`、`profile.npz`、`quality.json`、`result.json`（最终策略） | `best/` 子目录（含同样一套文件，训练期最优策略）；`diagnostics.npz`（奖励诊断、安全截断统计）、`evaluations.npz`（周期评估历史） |
| `dp_solve` | `run.json`、`profile.npz`、`quality.json` | — |
| `evaluation` | `run.json`、`profile.npz`、`quality.json`；评估 RL 策略时另需 `result.json` | — |

`profile.npz` 是 `SpeedProfile`（位置、速度、时间、分段加速度、牵引/悬浮能耗）；`quality.json` 是 `QualityReport`（指标、安全防护审计、完成/精确停站/安全/准点/可行判定）；`result.json`（仅 RL）记录终止原因、累计奖励、末状态与（训练运行）实际训练步数/预算达成情况。

## 论文实验与图表

`paper/` 下的仿真实验（空间步长消融、方法消融、DP/PIRS/实际运行三方对比、计划时间变化鲁棒性）及其结果复用规则详见 [`paper/README.md`](paper/README.md)。

## 测试与代码检查

```bash
uv run pytest                                   # 默认跳过标记为 slow、golden 的用例
uv run pytest -m slow                           # 耗时较长的用例
uv run python -m tests.golden.record            # 录制 golden 回归快照到 output/golden/
uv run pytest -m golden                         # 与已录制的快照比对
uv run ruff check src tests paper
uv run ruff format --check src tests paper
```

## Golden 回归快照

`tests/golden/` 下的测试用固定输入回放当前实现，比对结果与录制时保存的快照，用来发现"计算逻辑被无意改动"（另见[关键设计决定](#关键设计决定)中的版本号约定：论文结果复用依赖"改了计算逻辑就递增版本号"，golden 快照是唯一的自动检查）。

- **输入（入库）**：固定动作序列 `tests/golden/actions/*.npy`，由各用例的控制器生成后冻结；控制器与 `generate_actions` 只在显式传入 `--regenerate-actions <case>` 时才重新生成，默认直接回放已入库的序列。
- **输出（不入库）**：RL 固定动作轨迹、逐步奖励与观测、计划变更情形、DP 最优解，以及 RL/DP 用例的质量评估快照（`evaluation.quality.assess` 的全部指标、判定结果、越界与事件统计），写到仓库根目录下的 `output/golden/`（`.gitignore` 已排除 `output/`），并附 `manifest.json`（`mtto_version`、`git_commit`、`dirty`、录制时间、用例列表）。
- **录制**：`uv run python -m tests.golden.record [--case NAME] [--force] [--regenerate-actions NAME]`；已存在的快照不覆盖，除非 `--force`。
- **比对**：`uv run pytest -m golden`（`golden` 标记默认排除）。快照缺失时测试失败并提示先执行录制命令；比对失败时错误信息附带 `manifest.json` 中的 `mtto_version`、`git_commit`，便于判断是否改了计算逻辑却没递增版本号。默认的 `uv run pytest` 不依赖 `output/golden/`，在全新克隆上可直接通过。
- **使用约定**：在已知正确的干净提交上录制快照；修改代码后运行 `uv run pytest -m golden`；若差异是有意的计算逻辑变化，递增 `pyproject.toml` 中的 `project.version` 并重新录制（`--force`）。

`tests/paper/` 存放论文层（`paper/experiments`、`paper/analysis` 等）的测试。

## 扩展占位

`src/mtto/rl/morl/`、`app/ui/`、`app/api/` 目前只有占位 `README.md`，说明各自的扩展接缝与届时要做的事；详见其中内容与上文[扩展接缝](#扩展接缝)。
