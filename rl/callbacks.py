from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _record_namespace(self, attr_name: str, namespace: str) -> None:
        infos = self.training_env.get_attr(attr_name)
        if not infos:
            return

        aggregated_values: dict[str, list[float]] = {}
        for info in infos:
            if not isinstance(info, dict):
                continue
            for key, value in info.items():
                try:
                    scalar_value = float(value)
                except TypeError, ValueError:
                    continue
                aggregated_values.setdefault(key, []).append(scalar_value)

        for key, values in aggregated_values.items():
            if not values:
                continue
            self.logger.record(f"{namespace}/{key}", sum(values) / len(values))

    def _on_step(self) -> bool:
        # 以具名字典分层采集，避免在单一字典中混杂不同维度字段
        self._record_namespace(attr_name="rewards_info", namespace="rewards")
        self._record_namespace(attr_name="state_info", namespace="state")
        self._record_namespace(attr_name="constraint_info", namespace="constraint")
        self._record_namespace(attr_name="event_info", namespace="event")

        return super()._on_step()
