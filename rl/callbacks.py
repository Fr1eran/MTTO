from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 记录其他数据，如训练时得到的奖励
        # 从环境中直接读取 rewards_info
        rewards_infos = self.training_env.get_attr("rewards_info")

        # 假设是单环境训练或者只关注第一个环境
        if rewards_infos:
            rewards_info = rewards_infos[0]
            for key, value in rewards_info.items():
                self.logger.record(f"rewards/{key}", value)

        return super()._on_step()
