# MTTO

## 项目简介

MTTO 是一个面向中高速磁浮列车运行速度曲线优化的 Python 项目，包含动态规划（基线）与强化学习（主）两条优化链路。

## 目录结构

- `model/`: 轨道、车辆、能耗与防护核心模型。
- `rl/`: 强化学习环境与回调。
- `scripts/`: 可执行脚本入口。
- `tests/`: 单元测试。
- `data/`: 输入数据。
- `output/`: 训练与评估输出产物。

## 运行方式

推荐使用包化入口运行脚本：

- 训练：`python -m scripts.train_rl`
- 评估：`python -m scripts.evaluate_rl`
- 训练日志全维分析：`python -m scripts.analyze_training_data`
- 复现 DP：`python -m scripts.reproduce_dp`
- 绘制防护曲线：`python -m scripts.show_safeguard_curves`

### DP 复现实验脚本（reproduce_dp）

推荐命令：

- `python -m scripts.reproduce_dp`

状态转移图预计算支持通过 CLI 显式切换运行模式：

- `--precompute-mode {serial,parallel}`：预计算模式，默认 `serial`。
- `--precompute-workers N`：并行模式进程数；不指定时默认使用 `CPU-1`。
- `--precompute-chunk-size N`：并行模式每个任务块的阶段数；不指定时自动估计。
- `--mp-start-method {spawn,fork,forkserver}`：多进程启动方式；Windows 建议 `spawn`。
- `--hide-precompute-progress`：关闭预计算阶段进度条输出。

示例：

- 单进程（默认）：`python -m scripts.reproduce_dp --precompute-mode serial`
- 并行预计算（自动进程数）：`python -m scripts.reproduce_dp --precompute-mode parallel`
- 并行预计算（显式 4 进程 + 分块）：`python -m scripts.reproduce_dp --precompute-mode parallel --precompute-workers 4 --precompute-chunk-size 16 --mp-start-method spawn`
- 关闭预计算进度显示：`python -m scripts.reproduce_dp --precompute-mode parallel --hide-precompute-progress`

### 训练模式开关

训练脚本支持通过命令行参数 `--run-mode` 自动切换日志与分析开关：

- `tune`（默认）：启用 TensorBoard、训练回调采集、训练后自动分析。
- `reproduce`：关闭 TensorBoard、回调采集、训练后分析，优先训练效率。
- `eval`：与 `reproduce` 一样关闭训练日志相关开关（评估脚本也会显式禁用环境诊断采集）。

示例：

- 调优训练：`python -m scripts.train_rl --run-mode tune`
- 高效复现：`python -m scripts.train_rl --run-mode reproduce`
- 评估（不录视频）：`python -m scripts.evaluate_rl --no-record-video`

可选细粒度覆盖：

- `--enable-tb` / `--no-enable-tb`
- `--enable-callback` / `--no-enable-callback`
- `--enable-monitor` / `--no-enable-monitor`
- `--enable-env-diagnostics` / `--no-enable-env-diagnostics`
- `--enable-analysis` / `--no-enable-analysis`

分析输出目录可通过 `--analysis-output-root` 指定。

训练分析默认采用轻量输出，仅生成：

- `report.md`（核心性能、奖励质量、物理合规、演化趋势）
- `analysis_snapshot.json`（可复算结构化摘要）

默认不导出 CSV，也不保存 step/episode 原始快照，降低存储与大模型读取成本。
如需导出 CSV，可使用：

- `python -m scripts.analyze_training_data --export-csv`
- `python -m scripts.analyze_training_data --export-csv --include-snapshots`

常用训练参数也支持命令行覆盖：

- `--reward-discount`（默认 `0.99`）
- `--step-distance`（默认 `100.0`）
- `--num-envs`（默认 `1`，并行采样环境数量）
- `--vec-env-type`（默认 `subproc`，可选 `dummy|subproc`；仅 `num-envs>1` 时 `subproc` 生效）
- `--subproc-start-method`（默认 `spawn`，可选 `spawn|forkserver`）
- `--rollout-steps-per-update`（默认 `2048`，按所有并行环境合计）
- `--n-steps-per-env`（默认自动推导；优先级高于 `rollout-steps-per-update`）
- `--model-save-path`（默认 `output/optimal/rl/ppo_mtto`）
- `--vecnormalize-save-path`（默认 `output/optimal/rl/vecnormalize.pkl`）
- `--total-timesteps`（默认 `200000`）
- `--tensorboard-log-dir`（默认 `mtto_ppo_tensorboard_logs`）
- `--tb-log-name`（默认 `trainning_log`）
- `--log-interval`（仅在启用训练日志时生效；默认 `tune=1`，而 `reproduce/eval` 默认关闭日志，因此该参数默认被忽略）
- `--tb-sample-interval-steps`（默认 `1`，回调最小采样步长）
- `--env-diagnostics-interval-steps`（默认与 `--tb-sample-interval-steps` 一致，控制环境诊断快照输出频率）
- `--force-dump-interval-steps`（默认 `0` 关闭，>0 时按步长强制刷新缓冲中的 TensorBoard 事件）
- `--tb-batch-dump-records`（默认 `0` 关闭，>0 时按采样记录批量刷新缓冲中的 TensorBoard 事件）
- `--device`（默认 `cpu`）

训练后自动分析的采样质量闸门参数：

- `--analysis-min-points-per-10k-steps`（默认 `5.0`）
- `--analysis-min-unique-episodes`（默认 `100`）
- `--analysis-max-mean-step-gap`（默认 `2048.0`）
- `--analysis-sampling-quality-mode`（默认 `warn_only`，可选 `strict_fail`）

示例：

- 自定义训练核心参数：`python -m scripts.train_rl --run-mode tune --reward-discount 0.995 --step-distance 80 --total-timesteps 300000 --model-save-path output/optimal/rl/ppo_mtto_v2 --vecnormalize-save-path output/optimal/rl/vecnormalize_v2.pkl`

- 启用高密度采样并开启严格采样质量闸门：`python -m scripts.train_rl --run-mode tune --log-interval 1 --tb-sample-interval-steps 1 --analysis-sampling-quality-mode strict_fail`

- 基线单环境: `python -m scripts.train_rl --run-mode tune --num-envs 1 --vec-env-type dummy --tb-sample-interval-steps 1 --force-dump-interval-steps 0`

- 并行采样 + 高密度日志 + 批量落盘：`python -m scripts.train_rl --run-mode tune --num-envs 4 --vec-env-type subproc --tb-sample-interval-steps 10 --env-diagnostics-interval-steps 10 --tb-batch-dump-records 128 --force-dump-interval-steps 0`

## 测试

- 全量测试：`pytest`
- 指定目录：`pytest tests`