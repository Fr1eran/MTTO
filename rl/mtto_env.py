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
from rl.observation_builder import ObservationBuilder
from rl.operational_state import OperationalState, OperationalTransition
from rl.operational_stepper import OperationalStepper
from rl.reference_initial_state_provider import ReferenceInitialStateProvider
from rl.reward_calculator import RewardBreakdown, RewardCalculator, RewardConfig
from utils.indexing_utils import get_interval_index
from utils.plot_utils import set_chinese_font


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


class ConstraintInfo(TypedDict, total=False):
    margin_to_vmax_mps: float
    margin_to_vmin_mps: float
    violation_code: int
    speed_limit_segment: int


class EventInfo(TypedDict, total=False):
    episode_truncated_count: int
    episode_low_violation_count: int
    episode_high_violation_count: int


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
        max_step_distance: float,
        enable_diagnostics: bool = False,
        diagnostics_interval_steps: int = 1,
        enable_trajectory_tracking: bool = False,
        render_mode: str | None = None,
        use_animation: bool = False,
        reward_config: RewardConfig | None = None,
        reference_initial_state_provider: ReferenceInitialStateProvider | None = None,
    ) -> None:
        super().__init__()
        self.vehicle: VehicleInfo = vehicle
        self.track: TrackInfo = track
        self.safeguard_utility: SafeGuardUtility = safeguard_utility
        self.train_service: TrainService = train_service
        self.gamma: float = gamma
        self.max_step_distance: float = float(max_step_distance)
        self.stepper: OperationalStepper = OperationalStepper(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            max_step_distance_m=max_step_distance,
        )
        self.observation_builder: ObservationBuilder = ObservationBuilder(
            vehicle=vehicle,
            track=track,
            train_service=train_service,
            max_step_distance_m=max_step_distance,
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
            vehicle_max_speed_mps=vehicle.max_speed,
            reward_config=reward_config,
        )
        self.reward_config: RewardConfig = self.reward_calculator.reward_config
        self.reference_initial_state_provider: ReferenceInitialStateProvider | None = (
            reference_initial_state_provider
        )
        self._reference_context_sample_id: int | None = None
        self._reference_context_index: int | None = None
        self._reference_context_distribution_version: int | None = None
        self.state: OperationalState = self.stepper.reset()
        self.last_transition: OperationalTransition | None = None
        self.last_reward_breakdown: RewardBreakdown = RewardBreakdown()

        low = np.array([0, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0], dtype=np.float32)
        high = np.ones(12, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.enable_diagnostics: bool = enable_diagnostics
        self.diagnostics_interval_steps: int = max(1, int(diagnostics_interval_steps))
        self._collect_step_diagnostics: bool = False
        self.basic_info: BasicInfo = {}
        self.outcome_info: OutcomeInfo = {"terminated": False, "truncated": False}
        self.rewards_info: dict[str, float] = {}
        self.constraint_info: ConstraintInfo = {}
        self.event_info: EventInfo = {}
        self.episode_truncated_count: int = 0
        self.episode_low_violation_count: int = 0
        self.episode_high_violation_count: int = 0
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
        return self.observation_builder.build(self.state)

    def set_reference_initial_state_provider(
        self, provider: ReferenceInitialStateProvider | None
    ) -> None:
        """Attach or remove the optional reference-state reset provider."""
        self.reference_initial_state_provider = provider

    def set_reference_initial_state_distribution(
        self,
        weights: NDArray[np.floating] | list[float],
        *,
        version: int,
    ) -> None:
        """Update the provider distribution for resets after this call."""
        provider = self.reference_initial_state_provider
        if provider is None:
            raise RuntimeError("reference initial-state provider is not configured")
        provider.set_sampling_distribution(weights, version=version)

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
        provider = self.reference_initial_state_provider
        if provider is None:
            self.state = self.stepper.reset()
            self._reference_context_sample_id = None
            self._reference_context_index = None
            self._reference_context_distribution_version = None
        else:
            if seed is not None:
                provider.reseed(seed)
            reference_sample = provider.sample()
            self.state = reference_sample.runtime_state
            self._reference_context_sample_id = reference_sample.sample_id
            self._reference_context_index = reference_sample.reference_index
            self._reference_context_distribution_version = (
                reference_sample.distribution_version
            )
        self.last_transition = None
        self.last_reward_breakdown = RewardBreakdown()
        self.basic_info = {}
        self.outcome_info = {"terminated": False, "truncated": False}
        self.rewards_info = {}
        self.constraint_info = {}
        self.event_info = {}
        self.episode_truncated_count = self.episode_low_violation_count = (
            self.episode_high_violation_count
        ) = 0
        self._comfort_tav = self._comfort_sum_sq_delta_acc = 0.0
        self._comfort_exceedance_count = 0
        self._collect_step_diagnostics = False
        self._reset_trajectory()
        return self.observation_builder.build(self.state), {}

    @override
    def step(
        self,
        action: Any,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        acceleration = self.observation_builder.denormalize_action(float(action[0]))
        transition = self.stepper.advance(self.state, acceleration)
        reward = self.reward_calculator.calculate(transition)
        self.state, self.last_transition, self.last_reward_breakdown = (
            transition.next_state,
            transition,
            reward,
        )
        self.outcome_info = {
            "terminated": bool(transition.terminated),
            "truncated": bool(transition.truncated),
        }
        delta_acc = abs(
            transition.acceleration_mps2 - transition.previous_state.acceleration_mps2
        )
        self._comfort_tav += delta_acc
        self._comfort_sum_sq_delta_acc += delta_acc**2
        if delta_acc > self.train_service.max_acc_change:
            self._comfort_exceedance_count += 1
        self._record_trajectory()
        self._collect_step_diagnostics = (
            self.enable_diagnostics
            and self.state.step_count % self.diagnostics_interval_steps == 0
        )
        self._record_basic_info()
        info = self._get_basic_info()
        if self._reference_context_sample_id is not None:
            info.update(
                reference_context_sample_id=self._reference_context_sample_id,
                reference_context_index=self._reference_context_index,
                reference_context_distribution_version=(
                    self._reference_context_distribution_version
                ),
            )
        # Keep the terminal outcome available even when detailed diagnostics
        # are disabled (for example, vectorized evaluation rollouts).
        info.update(outcome=dict(self.outcome_info))
        if self._collect_step_diagnostics:
            self._record_step_diagnostics()
            info.update(
                rewards=dict(self.rewards_info),
                constraint=dict(self.constraint_info),
                event=dict(self.event_info),
            )
        return (
            self.observation_builder.build(self.state),
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

    def _record_step_diagnostics(self) -> None:
        vmax = self.state.max_speed_mps - self.state.speed_mps
        vmin = self.state.speed_mps - self.state.min_speed_mps
        if vmin < 0:
            self.episode_truncated_count += 1
            self.episode_low_violation_count += 1
        elif vmax < 0:
            self.episode_truncated_count += 1
            self.episode_high_violation_count += 1
        self.rewards_info = {
            "safety": self.last_reward_breakdown.safety,
            "energy": self.last_reward_breakdown.energy,
            "comfort": self.last_reward_breakdown.comfort,
            "stopping": self.last_reward_breakdown.stopping,
            "terminal_stopping": self.last_reward_breakdown.terminal_stopping,
            "punctuality": self.last_reward_breakdown.terminal_punctuality,
            "total": self.last_reward_breakdown.total,
        }
        segment = int(
            np.clip(
                get_interval_index(
                    self.state.position_m, self.track.speed_limit_intervals
                ),
                0,
                len(self.track.speed_limits) - 1,
            )
        )
        self.constraint_info = {
            "margin_to_vmax_mps": vmax,
            "margin_to_vmin_mps": vmin,
            "violation_code": int(self.last_transition.violation_code)
            if self.last_transition is not None
            else 0,
            "speed_limit_segment": segment,
        }
        self.event_info = {
            "episode_truncated_count": self.episode_truncated_count,
            "episode_low_violation_count": self.episode_low_violation_count,
            "episode_high_violation_count": self.episode_high_violation_count,
        }

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
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
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
