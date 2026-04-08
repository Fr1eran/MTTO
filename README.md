# MTTO

## 项目简介

MTTO 是一个面向中高速磁浮列车运行优化的 Python 项目，包含动态规划与强化学习两条优化链路。

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
- 复现 DP：`python -m scripts.reproduce_dp`
- 绘制防护曲线：`python -m scripts.show_safeguard_curves`

## 测试

- 全量测试：`pytest`
- 指定目录：`pytest tests`