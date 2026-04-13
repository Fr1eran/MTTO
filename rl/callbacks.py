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
        self._episode_ids_by_env: list[int] = []

    def _sync_episode_tracker(self, num_envs: int) -> None:
        if num_envs < 0:
            return
        if len(self._episode_ids_by_env) != num_envs:
            self._episode_ids_by_env = [0] * num_envs

    def _advance_episode_tracker(self) -> None:
        dones_raw = self.locals.get("dones")
        if dones_raw is None:
            return

        try:
            dones = list(dones_raw)
        except TypeError:
            return

        self._sync_episode_tracker(len(dones))
        for env_idx, done in enumerate(dones):
            if bool(done):
                self._episode_ids_by_env[env_idx] += 1

    def _get_namespace_payloads_from_locals(
        self, namespace: str
    ) -> list[dict[str, float]]:
        infos = self.locals.get("infos", [])
        if not isinstance(infos, (list, tuple)):
            return []

        self._sync_episode_tracker(len(infos))

        payloads: list[dict[str, float]] = []
        for info in infos:
            if not isinstance(info, dict):
                continue
            diagnostics = info.get("tb_diagnostics")
            if not isinstance(diagnostics, dict):
                continue
            namespace_payload = diagnostics.get(namespace)
            if isinstance(namespace_payload, dict):
                payloads.append(namespace_payload)

        return payloads

    def _enrich_state_payloads(
        self, payloads: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        enriched_payloads: list[dict[str, float]] = []
        for env_idx, payload in enumerate(payloads):
            payload_copy = dict(payload)
            if "episode_id" not in payload_copy:
                episode_id = (
                    self._episode_ids_by_env[env_idx]
                    if env_idx < len(self._episode_ids_by_env)
                    else 0
                )
                payload_copy["episode_id"] = float(episode_id)
            enriched_payloads.append(payload_copy)

        return enriched_payloads

    def _record_namespace(self, namespace: str) -> None:
        payloads = self._get_namespace_payloads_from_locals(namespace)
        if not payloads:
            return
        if namespace == "state":
            payloads = self._enrich_state_payloads(payloads)

        aggregated_values: dict[str, list[float]] = {}
        for payload in payloads:
            for key, value in payload.items():
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
            self._record_namespace(namespace="rewards")
            self._record_namespace(namespace="state")
            self._record_namespace(namespace="constraint")
            self._record_namespace(namespace="event")
            self._last_sample_step = current_step

        if (
            self.force_dump_interval_steps is not None
            and (current_step - self._last_dump_step) >= self.force_dump_interval_steps
        ):
            self.logger.dump(current_step)
            self._last_dump_step = current_step

        self._advance_episode_tracker()

        return True
