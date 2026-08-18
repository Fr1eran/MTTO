# MTTO 常用命令

以下命令默认从项目根目录执行：

```bash
cd /home/lopethos/gitprojects/MTTO
```

命令统一通过 `uv run` 使用项目虚拟环境。涉及 DSPDL 的命令假定对应的
DP 参考轨迹位于 `output/optimal/dp/465p0_0p1_uni30p0/`；如果实际目录不同，
只需替换 `--reference-curve-dir` 后的路径。

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
uv run python -m scripts.train_rl --dry-run
```

全量调优训练，每 12 个 rollouts 执行一次 best-eval：

```bash
uv run python -m scripts.train_rl \
  --output-root output/optimal/rl/tune/ \
  --schedule-time-s 465.0 \
  --step-distance 30.0 \
  --run-mode tune \
  --total-timesteps 1000000 \
  --num-envs 8 \
  --vec-env-type dummy \
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
  --run-mode monitor_best \
  --total-timesteps 1000000 \
  --num-envs 8 \
  --vec-env-type dummy \
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
  --run-mode reproduce \
  --total-timesteps 1000000 \
  --num-envs 8 \
  --vec-env-type dummy \
  --rollout-steps-per-update 8192 \
  --device cpu
```

启用 DSPDL 课程训练：

```bash
uv run python -m scripts.train_rl \
  --output-root output/optimal/rl/dspdl/ \
  --schedule-time-s 465.0 \
  --step-distance 30.0 \
  --reward-preset basic_safety \
  --curriculum-profile dspdl \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --run-mode monitor_best \
  --total-timesteps 1000000 \
  --num-envs 8 \
  --vec-env-type dummy \
  --rollout-steps-per-update 8192 \
  --evaluation-interval-rollouts 12 \
  --seed 11 \
  --device cpu
```

将上述命令中的课程配置改为以下值，可启用基于任务完成度的 DSPDL，同时保留旧
`dspdl` 配置用于对照：

```bash
  --curriculum-profile dspdl_completion
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
  --load-dir output/optimal/rl/tune/465p0_30p0/final/ \
  --dry-run
```

执行确定性评估并保存轨迹：

```bash
uv run python -m scripts.evaluate_rl \
  --load-dir output/optimal/rl/tune/465p0_30p0/final/ \
  --deterministic \
  --save-trajectory \
  --output-dir output/optimal/rl/evaluation/
```

评估并绘制运行时间序列：

```bash
uv run python -m scripts.evaluate_rl \
  --load-dir output/optimal/rl/tune/465p0_30p0/final/ \
  --deterministic \
  --plot-operation-time-series
```

查看训练期间按 rollout 评估得到的最佳轨迹：

```bash
uv run python -m scripts.show_rl_result \
  --curve-dir output/optimal/rl/tune/465p0_30p0/ \
  --trajectory-source best_rollouts
```

查看训练结束后的最终轨迹：

```bash
uv run python -m scripts.show_rl_result \
  --curve-dir output/optimal/rl/tune/465p0_30p0/ \
  --trajectory-source final
```

只预览将要加载的轨迹文件：

```bash
uv run python -m scripts.show_rl_result \
  --curve-dir output/optimal/rl/tune/465p0_30p0/ \
  --trajectory-source best \
  --dry-run
```

## 消融实验

预览步长消融运行矩阵：

```bash
uv run python -m scripts.run_step_distance_ablation train \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --num-envs 8 \
  --evaluation-interval-rollouts 12 \
  --dry-run
```

执行步长消融并展示结果：

```bash
uv run python -m scripts.run_step_distance_ablation train \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --num-envs 8 \
  --evaluation-interval-rollouts 12

uv run python -m scripts.run_step_distance_ablation show \
  --output-root output/optimal/rl/step_distance_ablation
```

预览奖励消融运行矩阵：

```bash
uv run python -m scripts.run_reward_ablation train \
  --num-envs 8 \
  --evaluation-interval-rollouts 12 \
  --dry-run
```

执行奖励消融并展示结果：

```bash
uv run python -m scripts.run_reward_ablation train \
  --num-envs 8 \
  --evaluation-interval-rollouts 12

uv run python -m scripts.run_reward_ablation show \
  --output-root output/optimal/rl/reward_ablation_safety
```

预览方法消融运行矩阵：

```bash
uv run python -m scripts.run_method_ablation train \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --num-envs 8 \
  --evaluation-interval-rollouts 12 \
  --dry-run
```

执行方法消融并展示结果：

```bash
uv run python -m scripts.run_method_ablation train \
  --reference-curve-dir output/optimal/dp/465p0_0p1_uni30p0/ \
  --num-envs 8 \
  --evaluation-interval-rollouts 12

uv run python -m scripts.run_method_ablation show \
  --output-root output/optimal/rl/method_ablation
```

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

运行环境吞吐基准：

```bash
uv run python -m scripts.benchmark_tune \
  --steps 8192 \
  --rollout-capacity 2048
```

## 对比与合规分析

对比 DP、RL 与实际运行速度曲线：

```bash
uv run python -m scripts.compare_speed_profiles \
  --dp-curve-dir output/optimal/dp/ \
  --rl-curve-dir output/optimal/rl/ \
  --trajectory-source best_rollouts
```

对比 DP 与 RL 的 SPS 合规性：

```bash
uv run python -m scripts.analyze_sps_compliance \
  --analysis-mode compare \
  --dp-curve-dir output/optimal/dp/ \
  --rl-curve-dir output/optimal/rl/ \
  --trajectory-source best_rollouts \
  --output-mode text+plot
```

将 SPS 分析同时保存为 JSON：

```bash
uv run python -m scripts.analyze_sps_compliance \
  --analysis-mode compare \
  --dp-curve-dir output/optimal/dp/ \
  --rl-curve-dir output/optimal/rl/ \
  --trajectory-source best_rollouts \
  --output-mode text+plot+json \
  --json-output-path output/analysis/sps_compliance.json
```

## 势函数可视化

```bash
uv run python -m scripts.show_potential_function --plot-type safety-speed
```

```bash
uv run python -m scripts.show_potential_function --plot-type safety-position
```

```bash
uv run python -m scripts.show_potential_function --plot-type stopping-heatmap
```

```bash
uv run python -m scripts.show_potential_function --plot-type stopping-slices
```

保存图片但不打开窗口：

```bash
uv run python -m scripts.show_potential_function \
  --plot-type guidance-wide \
  --output-file output/figures/guidance_wide.png \
  --no-show
```

## 测试与代码检查

运行完整测试集：

```bash
PYTHONPATH=. uv run pytest -q
```

运行 RL callback 和训练配置相关测试：

```bash
PYTHONPATH=. uv run pytest -q \
  tests/test_best_eval_callback.py \
  tests/test_experiment_utils.py \
  tests/test_train_rl_cli.py
```

运行 Ruff 静态检查：

```bash
uv run ruff check .
```

仅检查本次 rollout 评估改造相关文件：

```bash
uv run ruff check \
  rl/callbacks.py \
  rl/evaluation.py \
  rl/experiment_utils.py \
  scripts/train_rl.py
```
