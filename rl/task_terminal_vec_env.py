"""Preserve domain outcome flags while disabling bootstrap at task failures."""

from stable_baselines3.common.vec_env import VecEnvWrapper


class TaskTerminalVecEnv(VecEnvWrapper):
    """DummyVecEnv treats truncations as timeouts; internal MTTO ends are final."""

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        for done, info in zip(dones, infos, strict=True):
            if done and info.get("mtto_task_ended", False):
                info["TimeLimit.truncated"] = False
        return observations, rewards, dones, infos
