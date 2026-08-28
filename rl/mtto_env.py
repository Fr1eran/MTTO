import math
from typing import Any, TypedDict, cast, final, override

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.completion_critic import CompletionTrajectoryAccumulator
from rl.context_sampler import ContextSampler
from rl.dspdl import DSPDLStatisticsHub
from rl.observation_builder import ObservationBuilder
from rl.operational_state import OperationalState
from rl.operational_stepper import OperationalStepper
from rl.reward_calculator import RewardCalculator, RewardConfig
from rl.reward_diagnostics import RewardDiagnosticsAccumulator, RewardDiagnosticsBatch
from rl.safety_statistics import SafetyTruncationBatch, SafetyTruncationBuffer
from utils.plot_utils import sci_figure_size, set_chinese_font


class BasicInfo(TypedDict, total=False):
    position: float
    speed: float
    stopping_point_index: int
    operation_time: float
    redundant_operation_time: float
    energy_consumption: float
    comfort_tav: float
    comfort_er_pct: float
    comfort_rms: float


class OutcomeInfo(TypedDict):
    terminated: bool
    truncated: bool


@final
class MTTOEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gym adapter over the shared operational transition/reward pipeline."""

    metadata: dict[str, list[str]] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        vehicle: VehicleInfo,
        track: TrackInfo,
        safeguard_utility: SafeGuardUtility,
        train_service: TrainService,
        gamma: float,
        step_distance: float,
        compact_training_info: bool = False,
        enable_trajectory_tracking: bool = False,
        render_mode: str | None = None,
        use_animation: bool = False,
        reward_config: RewardConfig | None = None,
        stepper: OperationalStepper | None = None,
        context_sampler: ContextSampler | None = None,
        dspdl_statistics_hub: DSPDLStatisticsHub | None = None,
        curriculum_env_rank: int | None = None,
        completion_accumulator: CompletionTrajectoryAccumulator | None = None,
        safety_truncation_buffer: SafetyTruncationBuffer | None = None,
        reward_diagnostics_accumulator: RewardDiagnosticsAccumulator | None = None,
    ) -> None:
        super().__init__()
        if (dspdl_statistics_hub is None) != (curriculum_env_rank is None):
            raise ValueError(
                "DSPDL statistics hub and curriculum environment rank "
                "must be set together"
            )
        if dspdl_statistics_hub is not None and completion_accumulator is not None:
            raise ValueError(
                "traditional and completion DSPDL statistics are mutually exclusive"
            )
        if dspdl_statistics_hub is not None:
            if context_sampler is None:
                raise ValueError("DSPDL statistics require a context sampler")
            if not isinstance(curriculum_env_rank, (int, np.integer)):
                raise TypeError("curriculum_env_rank must be an integer")
            if not 0 <= int(curriculum_env_rank) < dspdl_statistics_hub.num_envs:
                raise IndexError("curriculum_env_rank is outside the statistics hub")
        self.vehicle: VehicleInfo = vehicle
        self.track: TrackInfo = track
        self.safeguard_utility: SafeGuardUtility = safeguard_utility
        self.train_service: TrainService = train_service
        self.gamma: float = gamma
        self.step_distance: float = float(step_distance)
        if stepper is not None:
            if (
                stepper.vehicle is not vehicle
                or stepper.track is not track
                or stepper.safeguard_utility is not safeguard_utility
                or stepper.train_service is not train_service
                or not math.isclose(stepper.step_distance_m, float(step_distance))
            ):
                raise ValueError("injected stepper does not match environment inputs")
            self.stepper = stepper
        else:
            self.stepper = OperationalStepper(
                vehicle=vehicle,
                track=track,
                safeguard_utility=safeguard_utility,
                train_service=train_service,
                step_distance_m=step_distance,
            )
        self.observation_builder: ObservationBuilder = ObservationBuilder(
            vehicle=vehicle,
            track=track,
            train_service=train_service,
            step_distance_m=step_distance,
            direction=self.stepper.direction,
            whole_distance_m=self.stepper.whole_distance_m,
            get_upper_speed_or_zero=self.stepper.get_upper_speed_or_zero,
        )
        self.reward_calculator: RewardCalculator = RewardCalculator(
            train_service,
            max_episode_steps=self.stepper.required_episode_steps,
            whole_distance_m=self.stepper.whole_distance_m,
            max_energy_consumption_kj=self.stepper.max_energy_consumption_kj,
            gamma=gamma,
            reward_config=reward_config,
        )
        self.reward_config: RewardConfig = self.reward_calculator.reward_config
        self.context_sampler = context_sampler
        self.dspdl_statistics_hub = dspdl_statistics_hub
        self.curriculum_env_rank = (
            int(curriculum_env_rank) if curriculum_env_rank is not None else None
        )
        self.completion_accumulator = completion_accumulator
        self.safety_truncation_buffer = safety_truncation_buffer
        self.reward_diagnostics_accumulator = reward_diagnostics_accumulator
        self._pending_dspdl_version: int | None = None
        self.state: OperationalState = self.stepper.reset()

        low = np.array([0, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0], dtype=np.float32)
        high = np.ones(12, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._observation_buffer: NDArray[np.float32] = np.empty(
            self.observation_space.shape, dtype=np.float32
        )
        self.compact_training_info: bool = bool(compact_training_info)
        self.basic_info: BasicInfo = {}
        self.outcome_info: OutcomeInfo = {"terminated": False, "truncated": False}
        self._comfort_tav: float = 0.0
        self._comfort_sum_sq_delta_acc: float = 0.0
        self._comfort_exceedance_count: int = 0
        self.enable_trajectory_tracking: bool = enable_trajectory_tracking
        self.trajectory_pos: list[float] | None = None
        self.trajectory_speed_mps: list[float] | None = None
        self.render_mode: str | None = render_mode
        self.use_animation: bool = use_animation
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.vehicle_dot: Line2D | None = None
        self.traj_line: Line2D | None = None
        self.animation: object | None = None
        self.animation_running: bool = False
        self.animation_interval: int = 15

    def change_schedule_time(self, new_schedule_time: float) -> NDArray[np.float32]:
        if new_schedule_time != self.train_service.schedule_time:
            self.train_service.schedule_time = new_schedule_time
            self.state = self.stepper.refresh_schedule_time(self.state)
        observation = self.observation_builder.build(
            self.state, out=self._observation_buffer
        )
        return observation.copy()

    def validate_dspdl_version(self, version: int) -> None:
        accumulator = self.completion_accumulator
        if self.context_sampler is None or accumulator is None:
            raise RuntimeError("DSPDL components are not configured")
        self._pending_dspdl_version = accumulator.validate_version_update(version)

    def commit_dspdl_version(self, version: int) -> None:
        accumulator = self.completion_accumulator
        if accumulator is None:
            raise RuntimeError("DSPDL accumulator is not configured")
        if self._pending_dspdl_version != int(version):
            raise ValueError("DSPDL version was not validated before commit")
        accumulator.switch_version(int(version))
        self._pending_dspdl_version = None

    def disable_dspdl_accumulator(self) -> None:
        completion_accumulator = self.completion_accumulator
        if completion_accumulator is not None:
            completion_accumulator.disable()

    def drain_completion_trajectories(self) -> dict[str, object]:
        accumulator = self.completion_accumulator
        if accumulator is None:
            raise RuntimeError("completion accumulator is not configured")
        return accumulator.drain()

    def drain_safety_truncations(self) -> SafetyTruncationBatch:
        buffer = self.safety_truncation_buffer
        if buffer is None:
            raise RuntimeError("safety truncation tracking is not configured")
        return buffer.drain()

    def drain_reward_diagnostics(
        self, *, finalize: bool = False
    ) -> RewardDiagnosticsBatch:
        accumulator = self.reward_diagnostics_accumulator
        if accumulator is None:
            raise RuntimeError("reward diagnostics are not configured")
        return accumulator.drain(finalize=finalize)

    def _reset_trajectory(self) -> None:
        if not self.enable_trajectory_tracking:
            self.trajectory_pos = self.trajectory_speed_mps = None
            return
        self.trajectory_pos = [self.state.position_m]
        self.trajectory_speed_mps = [abs(self.state.speed_mps)]

    def _record_trajectory(self) -> None:
        if self.enable_trajectory_tracking:
            assert (
                self.trajectory_pos is not None
                and self.trajectory_speed_mps is not None
            )
            self.trajectory_pos.append(self.state.position_m)
            self.trajectory_speed_mps.append(abs(self.state.speed_mps))

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        _ = super().reset(seed=seed, options=options)
        sampler = self.context_sampler
        if sampler is None:
            self.state = self.stepper.reset()
        else:
            if seed is not None:
                sampler.reseed(seed)
            context = sampler.sample()
            self.state = context.initial_state
            if self.dspdl_statistics_hub is not None:
                assert self.curriculum_env_rank is not None
                self.dspdl_statistics_hub.begin_episode(
                    env_rank=self.curriculum_env_rank,
                    context_index=context.context_index,
                    distribution_version=sampler.version,
                )
        self.basic_info = {}
        self.outcome_info = {"terminated": False, "truncated": False}
        self._comfort_tav = self._comfort_sum_sq_delta_acc = 0.0
        self._comfort_exceedance_count = 0
        self._reset_trajectory()
        observation = self.observation_builder.build(
            self.state, out=self._observation_buffer
        )
        if self.completion_accumulator is not None:
            self.completion_accumulator.begin_episode(observation)
        # Gym/VecEnv may retain this observation as terminal_observation while
        # immediately resetting the environment.  Do not expose the reusable
        # scratch buffer across that ownership boundary.
        return observation.copy(), {}

    @override
    def step(
        self,
        action: Any,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        acceleration = self.observation_builder.denormalize_action(float(action[0]))
        transition = self.stepper.advance(self.state, acceleration)
        reward = self.reward_calculator.calculate(transition)
        self.state = transition.next_state
        next_observation = self.observation_builder.build(
            self.state, out=self._observation_buffer
        )
        self.outcome_info = {
            "terminated": bool(transition.terminated),
            "truncated": bool(transition.truncated),
        }
        if self.dspdl_statistics_hub is not None:
            assert self.curriculum_env_rank is not None
            self.dspdl_statistics_hub.record_transition(
                self.curriculum_env_rank,
                reward.total,
                done=bool(transition.terminated or transition.truncated),
            )
        if self.completion_accumulator is not None:
            success_base, stopping_weight, punctuality_weight = (
                self.completion_accumulator.completion_weights
            )
            completion = self.reward_calculator.task_completion(
                terminated=bool(transition.terminated),
                truncated=bool(transition.truncated),
                stop_error_m=self.state.stop_error_m,
                operation_time_s=self.state.operation_time_s,
                success_base=success_base,
                stopping_weight=stopping_weight,
                punctuality_weight=punctuality_weight,
            )
            self.completion_accumulator.record_transition(
                next_observation,
                done=bool(transition.terminated or transition.truncated),
                completion=(
                    completion
                    if transition.terminated or transition.truncated
                    else None
                ),
            )
        if self.safety_truncation_buffer is not None:
            self.safety_truncation_buffer.record(
                position_m=self.state.position_m,
                violation_code=transition.violation_code,
                truncated=bool(transition.truncated),
            )
        if self.reward_diagnostics_accumulator is not None:
            self.reward_diagnostics_accumulator.record(
                reward,
                terminated=bool(transition.terminated),
                truncated=bool(transition.truncated),
                violation_code=int(transition.violation_code),
            )
        if not self.compact_training_info:
            delta_acc = abs(
                transition.acceleration_mps2
                - transition.previous_state.acceleration_mps2
            )
            self._comfort_tav += delta_acc
            self._comfort_sum_sq_delta_acc += delta_acc**2
            if delta_acc > self.train_service.max_acc_change:
                self._comfort_exceedance_count += 1
        self._record_trajectory()
        info: dict[str, object]
        if self.compact_training_info:
            info = {}
        else:
            self._record_basic_info()
            info = self._get_basic_info()
            info.update(outcome=dict(self.outcome_info))
        return (
            # See reset(): VecEnv may retain a terminal observation after this
            # method returns, so it must not alias the reusable scratch buffer.
            next_observation.copy(),
            reward.total,
            transition.terminated,
            transition.truncated,
            info,
        )

    def _record_basic_info(self) -> None:
        steps = max(self.state.step_count, 1)
        self.basic_info = {
            "energy_consumption": self.state.energy_consumption_kj,
            "operation_time": self.state.operation_time_s,
            "redundant_operation_time": self.state.redundant_operation_time_s,
            "position": self.state.position_m,
            "speed": self.state.speed_mps,
            "stopping_point_index": self.state.stopping_point_index,
            "comfort_tav": self._comfort_tav,
            "comfort_er_pct": self._comfort_exceedance_count / steps * 100.0,
            "comfort_rms": math.sqrt(self._comfort_sum_sq_delta_acc / steps),
        }

    def _get_basic_info(self) -> dict[str, object]:
        return {"basic": dict(self.basic_info)} if self.basic_info else {}

    @override
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render_mode."
            )
            return None
        return self._render(self.render_mode)

    def _render(self, mode: str):
        if self.fig is None:
            self._setup_figure(mode)
        if mode == "human":
            self._update_figure_data()
            assert self.fig is not None
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
            return None
        self._update_figure_data()
        assert self.fig is not None
        canvas = cast(FigureCanvasAgg, self.fig.canvas)
        canvas.draw()
        w, h = canvas.get_width_height()
        return (
            np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            .reshape((h, w, 4))[:, :, :3]
            .copy()
        )

    def _setup_figure(self, mode: str) -> None:
        set_chinese_font()
        if mode == "human":
            _ = plt.ion()
        self.fig, self.ax = plt.subplots(
            figsize=sci_figure_size(columns=2, height_in=3.8)
        )
        _ = self.fig.suptitle("磁悬浮列车智能体训练过程", fontsize=14)
        margin = self.stepper.whole_distance_m * 0.1
        _ = self.ax.set_xlim(
            min(self.train_service.start_position, self.train_service.target_position)
            - margin,
            max(self.train_service.start_position, self.train_service.target_position)
            + margin,
        )
        _ = self.ax.set_ylim(0, 600)
        self.safeguard_utility.render(ax=self.ax)
        (self.vehicle_dot,) = self.ax.plot([], [], "g*", markersize=8, label="列车")
        (self.traj_line,) = self.ax.plot([], [], "b-", lw=2, label="轨迹")
        _ = self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _update_figure_data(self) -> None:
        assert self.vehicle_dot is not None
        self.vehicle_dot.set_data([self.state.position_m], [self.state.speed_mps * 3.6])
        if self.trajectory_pos and self.trajectory_speed_mps:
            assert self.traj_line is not None
            self.traj_line.set_data(
                self.trajectory_pos, [x * 3.6 for x in self.trajectory_speed_mps]
            )

    @override
    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
