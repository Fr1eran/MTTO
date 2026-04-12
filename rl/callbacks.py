from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(
        self,
        verbose: int = 0,
        min_tb_sample_interval_steps: int = 1,
        force_dump_interval_steps: int | None = None,
    ):
        super().__init__(verbose)
        self.min_tb_sample_interval_steps = max(1, int(min_tb_sample_interval_steps))
        self.force_dump_interval_steps = (
            None
            if force_dump_interval_steps is None or int(force_dump_interval_steps) <= 0
            else int(force_dump_interval_steps)
        )
        self._last_sample_step: int = -self.min_tb_sample_interval_steps
        self._last_dump_step: int = 0

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
        current_step = int(self.num_timesteps)

        # 控制每次记录的最小步长间隔，避免高频写入造成 I/O 开销。
        should_sample = (
            current_step - self._last_sample_step
        ) >= self.min_tb_sample_interval_steps

        if should_sample:
            self._record_namespace(attr_name="rewards_info", namespace="rewards")
            self._record_namespace(attr_name="state_info", namespace="state")
            self._record_namespace(attr_name="constraint_info", namespace="constraint")
            self._record_namespace(attr_name="event_info", namespace="event")
            self._last_sample_step = current_step

        if (
            self.force_dump_interval_steps is not None
            and (current_step - self._last_dump_step) >= self.force_dump_interval_steps
        ):
            self.logger.dump(current_step)
            self._last_dump_step = current_step

        return True
