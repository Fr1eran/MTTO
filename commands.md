# MTTO 常用命令

以下命令默认从项目根目录执行：

```bash
cd /home/lopethos/gitprojects/MTTO
export UV_CACHE_DIR=/tmp/mtto-uv-cache
export MPLCONFIGDIR=/tmp/mtto-mpl-cache
export PYTHONPATH=.
```

命令统一通过 `uv run` 使用项目虚拟环境。常规 DSPL 命令假定对应的
DP 参考轨迹位于 `output/optimal/dp/465p0_0p1_uni30p0/`；如果实际目录不同，
只需替换 `--reference-curve-dir` 后的路径。步长消融和方法消融使用 10 m
参考曲线 `output/optimal/dp/465p0_0p1_uni10p0/`；方法消融的 RL 环境步长仍为 30 m，
仅 DSPL/context 的来源轨迹改用 10 m DP。

主训练入口和通用训练参数默认使用 `dspl`（DSPL）。课程 profile 启用时
仍必须显式提供与任务匹配的 DP 参考目录；消融或基线实验可显式指定
`--curriculum-profile none`。旧任务完成度课程的输出目录和 manifest 只作为历史材料保存，
不会被当前实验脚本自动迁移或续跑。

## 环境准备

安装或同步依赖：

```bash
uv sync
```

查看主训练入口的完整参数：

```bash
uv run python -m scripts.train_rl --help
```

## DP 参考轨迹

生成 465 秒、30 米等间距的 DP 参考轨迹：

```bash
uv run python -m scripts.reproduce_dp \
  --schedule-time-s 465.0 \
  --delta-speed-mps 0.1 \
  --stage-division uniform \
  --uniform-step-size 30.0 \
  --precompute-mode parallel \
  --precompute-workers 4
```

生成变间距 DP 参考轨迹：

```bash
uv run python -m scripts.reproduce_dp \
  --schedule-time-s 465.0 \
  --delta-speed-mps 0.1 \
  --stage-division variable \
  --sub-stage-count 30 \
  --precompute-mode parallel \
  --precompute-workers 4 \
  --precompute-chunk-size 15
```

查看 DP 轨迹：

```bash
uv run python -m scripts.show_dp_result \
  --curve-dir output/optimal/dp/465p0_0p1_uni30p0/
```

## RL 训练

先预览默认训练配置，不创建环境、不启动训练：

```bash
uv run python -m scripts.train_rl \
  --curriculum-profile dspl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --dry-run
```

全量调优训练，每 12 个 rollouts 执行一次 best-eval：

```bash
uv run python -m scripts.train_rl \
  --output-root output/optimal/rl/tune/ \
  --schedule-time-s 465.0 \
  --step-distance 30.0 \
  --curriculum-profile dspl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --run-mode tune \
  --training-episodes 5000 \
  --num-envs 8 \
  --rollout-steps-per-update 8192 \
  --evaluation-interval-rollouts 12 \
  --evaluation-deterministic \
  --device cpu
```

低开销训练，仅保留基础监控和 best-eval：

```bash
uv run python -m scripts.train_rl \
  --output-root output/optimal/rl/monitor_best/ \
  --schedule-time-s 465.0 \
  --step-distance 30.0 \
  --curriculum-profile dspl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --run-mode monitor_best \
  --training-episodes 5000 \
  --num-envs 8 \
  --rollout-steps-per-update 8192 \
  --evaluation-interval-rollouts 12 \
  --evaluation-deterministic \
  --device cpu
```

高效复现训练，不执行训练期 best-eval：

```bash
uv run python -m scripts.train_rl \
  --output-root output/optimal/rl/reproduce/ \
  --schedule-time-s 465.0 \
  --step-distance 30.0 \
  --curriculum-profile dspl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --run-mode reproduce \
  --training-episodes 5000 \
  --num-envs 8 \
  --rollout-steps-per-update 8192 \
  --device cpu
```

论文主方法使用 DSPL。`train_rl` 的默认课程配置已经是
`dspl`，也可以在正式命令中显式写出：

```bash
  --curriculum-profile dspl
```

使用 CUDA 训练时，将上述命令末尾的 `--device cpu` 改为 `--device cuda`。

查看 TensorBoard：

```bash
uv run tensorboard --logdir mtto_ppo_tb_logs
```

## RL 模型评估与轨迹查看

预览模型加载路径和评估配置：

```bash
uv run python -m scripts.evaluate_rl \
  --model-dir output/optimal/rl/tune/465p0_30p0/final/ \
  --dry-run
```

执行确定性评估并保存轨迹：

```bash
uv run python -m scripts.evaluate_rl \
  --model-dir output/optimal/rl/tune/465p0_30p0/final/ \
  --deterministic \
  --save-trajectory \
  --output-dir output/optimal/rl/evaluation/
```

评估并绘制运行时间序列：

```bash
uv run python -m scripts.evaluate_rl \
  --model-dir output/optimal/rl/tune/465p0_30p0/final/ \
  --deterministic \
  --plot-operation-time-series
```

查看训练期间按 rollout 评估得到的最佳轨迹：

```bash
uv run python -m scripts.show_rl_result \
  --model-dir output/optimal/rl/tune/465p0_30p0/best/
```

查看训练结束后的最终轨迹：

```bash
uv run python -m scripts.show_rl_result \
  --model-dir output/optimal/rl/tune/465p0_30p0/final/
```

只预览将要加载的轨迹文件：

```bash
uv run python -m scripts.show_rl_result \
  --model-dir output/optimal/rl/tune/465p0_30p0/best/ \
  --dry-run
```

## 消融实验

训练入口通过 `--budget-mode` 选择 `completed_episodes` 或 `environment_steps`。
前者使用 `--training-episodes`，后者使用 `--training-rollouts`。预算信息只记录在
`metadata.json` 的 `training_budget.mode` 及其对应字段中。
该预算接口已更新，旧版本 manifest 不能用于 `--resume`；请为新实验使用新的输出根目录。
两类消融脚本均将矩阵状态写入各自输出根目录的 `manifest.json`。需要断点恢复时显式追加
`--resume`；runner 只会跳过 canonical 产物完整且完成回合预算已达标的运行，并在首个失败运行处停止，
其余运行保留为 `pending`。如果输出目录已有 manifest 但未指定 `--resume`，新训练会拒绝覆盖；
确认要重新开始时使用 `--force-new`，旧 manifest 会先备份为带 UTC 时间戳的文件。resume 只检查
canonical 产物，不会在检查阶段自动复制旧版产物。
评估指标使用严格 schema v2：安全必须同时满足
`min_safety_margin_mps >= -1e-6` 和 `safety_violation_count == 0`，
`metrics.json` 显式保存 `safe`、`feasible` 和 `selection_comparison_key`。

预览步长消融运行矩阵：

```bash
uv run python -m scripts.run_step_distance_ablation train \
  --output-root output/paper_experiment/01_step_distance/20260910_01 \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni10p0/ \
  --num-envs 8 \
  --rollout-steps-per-update 8192 \
  --evaluation-interval-episodes 100 \
  --dry-run
```

执行步长消融并展示结果：

```bash
uv run python -m scripts.run_step_distance_ablation train \
  --output-root output/paper_experiment/01_step_distance/20260910_01 \
  --num-envs 8 \
  --training-episodes 4000 \
  --evaluation-interval-episodes 100

uv run python -m scripts.run_step_distance_ablation show \
  --output-root output/paper_experiment/01_step_distance/20260910_01
```

步长消融以每个种子相同的 4000 个完整物理行程为预算。按回合阈值触发的评估会等到
下一次 PPO 更新后的 rollout 起点，因此 `evaluations.npz` 中的实际完成回合数可能略大于
规则目标；主图按规则目标精确对齐，先在每个种子内做 5 点尾随平均，再跨 5 个种子绘制
均值与样本标准差带。4000 回合不做周期评估，训练结束后仍执行一次 `final` 独立评估。

预览方法消融运行矩阵：

```bash
uv run python -m scripts.run_method_ablation train \
  --output-root output/paper_experiment/02_method_ablation/20260910_01 \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni10p0/ \
  --num-envs 8 \
  --evaluation-interval-rollouts 12 \
  --dry-run
```

执行方法消融并展示结果：

```bash
uv run python -m scripts.run_method_ablation train \
  --output-root output/paper_experiment/02_method_ablation/20260910_01 \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni10p0/ \
  --num-envs 8 \
  --evaluation-interval-rollouts 12

uv run python -m scripts.run_method_ablation show \
  --output-root output/paper_experiment/02_method_ablation/20260910_01 \
  --figure-output-dir output/paper_experiment/02_method_ablation/20260910_01 \
  --selection-output-file output/paper_experiment/02_method_ablation/20260910_01/selected_policy.json
```

两类曲线都先在每个种子内做 5 个周期评估点的尾随平滑，再跨
5 个种子计算均值和样本标准差；失败评估不剔除。

## 训练分析与性能诊断

分析 TensorBoard 日志和训练二进制诊断数据：

```bash
uv run python -m scripts.analyze_training_data \
  --log-root mtto_ppo_tb_logs \
  --final-output-dir output/optimal/rl/tune/465p0_30p0/final/ \
  --output-root mtto_train_reports \
  --rollout-steps-per-update 8192 \
  --sampling-quality-mode warn_only
```

预览分析配置：

```bash
uv run python -m scripts.analyze_training_data \
  --final-output-dir output/optimal/rl/tune/465p0_30p0/final/ \
  --dry-run
```

## 对比与合规分析

先将实际选中的 `best/` 或 `final/` 目录设为复用变量：

```bash
RL_MODEL_DIR="output/optimal/rl/tune/465p0_30p0/best"
```

对比 DP、RL 与实际运行速度曲线：

```bash
uv run python -m scripts.compare_speed_profiles \
  --dp-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --rl-model-dir "$RL_MODEL_DIR"
```

对比 DP 与 RL 的 SPS 合规性：

```bash
uv run python -m scripts.analyze_sps_compliance \
  --analysis-mode compare \
  --dp-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --rl-model-dir "$RL_MODEL_DIR" \
  --output-mode text+plot
```

将 SPS 分析同时保存为 JSON：

```bash
uv run python -m scripts.analyze_sps_compliance \
  --analysis-mode compare \
  --dp-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --rl-model-dir "$RL_MODEL_DIR" \
  --output-mode text+plot+json \
  --json-output-path output/analysis/sps_compliance.json
```

`--rl-model-dir` 必须指向直接包含 `policy.zip`、`trajectory.npz`、
`metrics.json` 和 `metadata.json` 的 `best/` 或 `final/` 目录。方法消融后可从
`selected_policy.json` 的 `selected.model_dir` 获取代表策略目录；该路径
相对于选择文件所在目录。

## 运行时间突变实验

```bash
uv run python -m scripts.run_schedule_time_change evaluate \
  --method-ablation-dir output/paper_experiment/02_method_ablation/20260910_01 \
  --output-dir output/paper_experiment/04_schedule_time_change/20260910_01 \
  --change-distance-m 8000 \
  --delta-times-s=-30,0,30 \
  --deterministic \
  --device cpu

uv run python -m scripts.run_schedule_time_change show \
  --load-dir output/paper_experiment/04_schedule_time_change/20260910_01 \
  --save-figure \
  --no-show
```

## 防护曲线计算

预览有效参数、危险点数量和计划输出文件，不执行计算：

```bash
uv run python -m scripts.calc_and_save_safeguard_curves --dry-run
```

使用默认参数重新生成防护曲线。默认目录已有产物，因此需要显式允许覆盖：

```bash
uv run python -m scripts.calc_and_save_safeguard_curves --force
```

使用更细的距离步长并写入独立目录：

```bash
uv run python -m scripts.calc_and_save_safeguard_curves \
  --output-dir output/safeguardcurves_0p5m \
  --distance-step-m 0.5
```

查看车辆、安全误差和延时等完整参数：

```bash
uv run python -m scripts.calc_and_save_safeguard_curves --help
```

## 线路环境与防护曲线可视化

展示综合环境视图（防护曲线 + 轨道坡度）：

```bash
uv run python -m scripts.show_env_data --view overview
```

展示全量安全防护曲线：

```bash
uv run python -m scripts.show_env_data --view full-curves
```

展示危险速度域与交叉点：

```bash
uv run python -m scripts.show_env_data --view danger-region
```

保存所有视图至指定目录且不弹出交互窗口：

```bash
uv run python -m scripts.show_env_data \
  --view all \
  --output-dir output/figures \
  --no-show
```

## 势函数可视化

完整线路准点势函数：

```bash
uv run python -m scripts.show_potential_function --plot-type punctuality
```

安全势与准点势双栏联合图：

```bash
uv run python -m scripts.show_potential_function \
  --plot-type safety-punctuality \
  --schedule-time-s 465
```

```bash
uv run python -m scripts.show_potential_function --plot-type safety-speed
```

保存图片但不打开窗口：

```bash
uv run python -m scripts.show_potential_function \
  --plot-type safety-punctuality \
  --output-dir output/figures \
  --no-show
```

## 终端评分函数可视化

展示停站与准点综合评分曲线：

```bash
uv run python -m scripts.show_score_function --plot combined
```

展示停站误差评分曲线：

```bash
uv run python -m scripts.show_score_function --plot stopping
```

展示准点时间误差评分曲线：

```bash
uv run python -m scripts.show_score_function --plot punctuality
```

保存为图片且不弹出交互窗口：

```bash
uv run python -m scripts.show_score_function \
  --plot combined \
  --output-dir output/figures \
  --no-show
```

## 测试与代码检查

运行完整测试集：

```bash
uv run pytest -q
```

运行 RL callback 和训练配置相关测试：

```bash
uv run pytest -q \
  tests/test_best_eval_callback.py \
  tests/test_experiment_utils.py \
  tests/test_train_rl_cli.py
```

运行 Ruff 静态检查：

```bash
uv run ruff check .
uv run ruff format --check
```

仅检查本次 rollout 评估改造相关文件：

```bash
uv run ruff check \
  rl/callbacks.py \
  rl/evaluation.py \
  rl/experiment_utils.py \
  scripts/train_rl.py
```
