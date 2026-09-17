from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rl.experiment_utils import (
    DEFAULT_COMFORT_REWARD_SCALE,
    DEFAULT_DEVICE,
    DEFAULT_ENERGY_REWARD_SCALE,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_PRESET_NAME,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    RUN_METADATA_FILENAME,
    build_default_training_args,
    build_reward_config,
    build_rl_trajectory_comparison_key,
    build_run_metadata,
    load_run_metadata,
    render_rl_curve_on_axes,
    resolve_output_dir,
    resolve_reward_preset,
    resolve_tb_log_name,
    resolve_training_run_spec,
    reward_preset_names,
    save_run_metadata,
)


def test_default_training_args_use_shared_vector_environment_defaults() -> None:
    args = build_default_training_args()

    assert DEFAULT_NUM_ENVS == 8
    assert args.num_envs == DEFAULT_NUM_ENVS
    assert not hasattr(args, "vec_env_type")
    assert args.rollout_steps_per_update == DEFAULT_ROLLOUT_STEPS_PER_UPDATE
    assert args.device == DEFAULT_DEVICE
    assert args.reward_preset == "basic_safety_punctuality"


def test_render_rl_curve_wrapper_supplies_rl_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "rl.experiment_utils.render_trajectory_on_axes",
        lambda **kwargs: captured.update(kwargs),
    )
    position = np.asarray([0.0, 10.0])
    speed = np.asarray([0.0, 1.0])
    metrics = {"trajectory_source": "final"}

    render_rl_curve_on_axes(
        ax="axis",
        pos_arr=position,
        speed_arr=speed,
        metrics=metrics,
        no_safeguard=False,
        factor=0.97,
        render_endpoints=False,
    )

    assert captured["ax"] == "axis"
    assert captured["pos_arr"] is position
    assert captured["speed_arr"] is speed
    assert captured["metrics"] is metrics
    assert captured["curve_color"] == "blue"
    assert captured["curve_label"] == "RL final trajectory"
    assert captured["no_safeguard"] is False
    assert captured["factor"] == pytest.approx(0.97)
    assert captured["render_endpoints"] is False


def test_learning_rate_anneals_by_completed_episodes_not_sb3_progress() -> None:
    from rl.callbacks import CompletedEpisodeProgress
    from rl.experiment_utils import _completed_episode_cosine_annealing_schedule

    progress = CompletedEpisodeProgress(target_episodes=5000)
    schedule = _completed_episode_cosine_annealing_schedule(progress)
    assert schedule(0.0) == pytest.approx(3e-4)
    assert schedule(1.0) == pytest.approx(3e-4)

    progress.completed_episodes = 2500
    assert schedule(0.0) == pytest.approx((3e-4 + 1e-5) / 2.0)
    progress.completed_episodes = 5000
    assert schedule(1.0) == pytest.approx(1e-5)
    progress.completed_episodes = 6000
    assert schedule(0.5) == pytest.approx(1e-5)


def test_learning_rate_anneals_by_environment_step_progress() -> None:
    from rl.experiment_utils import _environment_step_cosine_annealing_schedule

    schedule = _environment_step_cosine_annealing_schedule()
    assert schedule(1.0) == pytest.approx(3e-4)
    assert schedule(0.5) == pytest.approx((3e-4 + 1e-5) / 2.0)
    assert schedule(0.0) == pytest.approx(1e-5)


def test_environment_step_budget_resolves_to_complete_rollouts() -> None:
    args = build_default_training_args()
    args.budget_mode = "environment_steps"
    args.training_rollouts = 500

    spec = resolve_training_run_spec(args)

    assert spec.training_episodes is None
    assert spec.training_rollouts == 500
    assert spec.total_timesteps == 4_096_000
    budget = spec.run_metadata.training_budget
    assert budget is not None
    assert budget.mode == "environment_steps"
    assert budget.training_rollouts == 500
    assert budget.effective_training_episodes is None


def test_reward_preset_names_include_pirs_component_profiles() -> None:
    assert reward_preset_names() == (
        "basic",
        "basic_safety",
        "basic_punctuality",
        "basic_safety_punctuality",
    )


@pytest.mark.parametrize(
    "profile_name",
    (
        "basic_stopping",
        "basic_safety_stopping",
        "all",
        "full",
        "full_shaping",
        "basic_safety_stopping_punctuality",
    ),
)
def test_removed_reward_profiles_are_rejected(profile_name: str) -> None:
    with pytest.raises(ValueError):
        _ = resolve_reward_preset(profile_name)


def test_reward_presets_keep_energy_comfort_and_toggle_potential_shaping() -> None:
    expected_flags = {
        "basic": False,
        "basic_safety": True,
    }
    for profile_name, expected in expected_flags.items():
        reward_config = build_reward_config(profile_name)
        assert reward_config.energy_reward_scale == DEFAULT_ENERGY_REWARD_SCALE
        assert reward_config.comfort_reward_scale == DEFAULT_COMFORT_REWARD_SCALE
        assert reward_config.enable_potential_safety is expected


def test_reward_preset_owns_the_runtime_config() -> None:
    preset = resolve_reward_preset("basic_safety")

    assert preset.config is build_reward_config("basic_safety")
    assert preset.enabled_shaping_components() == ("safety",)


def test_reward_metadata_contains_no_global_scaling_fields() -> None:
    metadata = resolve_reward_preset("basic_safety").to_metadata()

    assert all("normalization" not in key for key in metadata)
    assert all("normalization" not in key for key in metadata["reward_config"])


def test_resolve_output_dir_scopes_default_reward() -> None:
    output_dir = resolve_output_dir(
        output_root="output/optimal/rl",
        schedule_time_s=430.0,
        step_distance=100.0,
        reward_preset_name=DEFAULT_REWARD_PRESET_NAME,
    )

    assert Path(output_dir).name == "430p0_100p0__basic_safety_punctuality"


def test_resolve_output_dir_scopes_reward_and_experiment_tag() -> None:
    output_dir = resolve_output_dir(
        output_root="output/optimal/rl",
        schedule_time_s=430.0,
        step_distance=100.0,
        reward_preset_name="basic",
        experiment_tag="Trial A",
    )

    assert Path(output_dir).name == "430p0_100p0__basic__trial_a"


def test_resolve_tb_log_name_generates_experiment_scoped_name() -> None:
    tb_log_name = resolve_tb_log_name(
        tb_log_name=None,
        run_mode="monitor_best",
        schedule_time_s=430.0,
        step_distance=100.0,
        reward_preset_name=DEFAULT_REWARD_PRESET_NAME,
        experiment_tag=None,
    )

    assert (
        tb_log_name == "train_log__monitor_best__430p0_100p0__basic_safety_punctuality"
    )


def test_load_run_metadata_falls_back_to_parent_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "430p0_100p0__basic"
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True)

    expected_metadata = build_run_metadata(
        reward_preset=resolve_reward_preset("basic"),
        schedule_time_s=430.0,
        step_distance=100.0,
        reward_discount=0.998,
        run_mode="monitor_best",
        tb_log_name="train_log__monitor_best__430p0_100p0__basic",
    )
    metadata_path = save_run_metadata(run_dir, expected_metadata)

    assert Path(metadata_path).name == RUN_METADATA_FILENAME
    assert load_run_metadata(run_dir) == expected_metadata
    assert load_run_metadata(final_dir) == expected_metadata


def test_build_rl_trajectory_comparison_key_uses_selection_key() -> None:
    success_high_energy = {
        "selection_comparison_key": [1.0, 1.0, 0.0, 1.0, 0.0, -8_000.0],
    }
    success_low_energy = {
        "selection_comparison_key": [1.0, 1.0, 0.0, 1.0, 0.0, -4_000.0],
    }
    failure_high_reward = {
        "selection_comparison_key": [0.0, 999.0],
    }

    assert build_rl_trajectory_comparison_key(
        success_low_energy
    ) > build_rl_trajectory_comparison_key(success_high_energy)
    assert build_rl_trajectory_comparison_key(
        success_low_energy
    ) > build_rl_trajectory_comparison_key(failure_high_reward)


def test_build_rl_trajectory_comparison_key_requires_new_metrics() -> None:
    with pytest.raises(ValueError, match="schema v2"):
        _ = build_rl_trajectory_comparison_key(
            {
                "success": True,
                "total_energy_j": 4_000.0,
                "stop_error_m": 0.2,
                "time_error_s": 1.0,
                "total_reward": 10.0,
            }
        )


def test_add_panel_label_places_text_on_axes() -> None:
    from utils.plot_utils import add_panel_label

    fig, ax = plt.subplots()
    text = add_panel_label(ax=ax, label="(a)")

    assert text.get_text() == "(a)"
    assert text.get_position() == (0.02, 0.98)
    assert text.get_ha() == "left"
    assert text.get_va() == "top"

    plt.close(fig)


def test_resolve_survival_reward_scale_fallback_on_negative_or_invalid() -> None:
    from rl.experiment_utils import (
        DEFAULT_SURVIVAL_REWARD_SCALE,
        resolve_survival_reward_scale,
        reward_config_parameters,
    )

    assert resolve_survival_reward_scale(None) == DEFAULT_SURVIVAL_REWARD_SCALE
    assert resolve_survival_reward_scale(50.0) == 50.0
    assert resolve_survival_reward_scale(0.0) == 0.0
    assert resolve_survival_reward_scale(-10.0) == DEFAULT_SURVIVAL_REWARD_SCALE
    assert resolve_survival_reward_scale(float("nan")) == DEFAULT_SURVIVAL_REWARD_SCALE
    assert resolve_survival_reward_scale(float("inf")) == DEFAULT_SURVIVAL_REWARD_SCALE

    cfg = build_reward_config("basic_safety", survival_reward_scale=-5.0)
    assert cfg.survival_reward_scale == DEFAULT_SURVIVAL_REWARD_SCALE

    cfg_custom = build_reward_config("basic", survival_reward_scale=30.0)
    assert cfg_custom.survival_reward_scale == 30.0

    d = reward_config_parameters(cfg_custom)
    assert d["energy_reward_scale"] == DEFAULT_ENERGY_REWARD_SCALE
    assert d["comfort_reward_scale"] == DEFAULT_COMFORT_REWARD_SCALE
    assert d["survival_reward_scale"] == 30.0
    assert d["enable_potential_safety"] is False
    assert "enable_energy" not in d
    assert "enable_comfort" not in d


def test_punctuality_potential_parameters_are_not_runtime_configurable() -> None:
    defaults = build_reward_config("basic_safety_punctuality")
    assert defaults.enable_potential_punctuality
    assert not hasattr(defaults, "punctuality_potential_scale")
    assert not hasattr(defaults, "punctuality_potential_sigma_s")


def test_derive_training_budget_rules() -> None:
    from rl.experiment_utils import _derive_training_budget, resolve_training_run_spec

    # 1. 7000 回合、8 环境的有效目标仍为 7000
    effective, max_steps, derived_timesteps = _derive_training_budget(
        training_episodes=7000,
        num_envs=8,
        step_distance=30.0,
        rollout_steps_per_update=8192,
        schedule_time_s=465.0,
    )
    assert effective == 7000
    assert derived_timesteps % 8192 == 0
    assert derived_timesteps >= effective * max_steps

    # 2. 非整除回合数向上取整
    effective_odd, _, _ = _derive_training_budget(
        training_episodes=7001,
        num_envs=8,
        step_distance=30.0,
        rollout_steps_per_update=8192,
        schedule_time_s=465.0,
    )
    assert effective_odd == 7008

    # 3. 10m 与 100m 产生不同的最大单回合步数及内部 SB3 总步数
    _, max_steps_10, timesteps_10 = _derive_training_budget(
        training_episodes=7000,
        num_envs=8,
        step_distance=10.0,
        rollout_steps_per_update=8192,
        schedule_time_s=465.0,
    )
    _, max_steps_100, timesteps_100 = _derive_training_budget(
        training_episodes=7000,
        num_envs=8,
        step_distance=100.0,
        rollout_steps_per_update=8192,
        schedule_time_s=465.0,
    )
    assert max_steps_10 > max_steps_100
    assert timesteps_10 > timesteps_100
    assert timesteps_10 % 8192 == 0
    assert timesteps_100 % 8192 == 0

    # 4. resolve_training_run_spec 快照正确传递 training_budget
    args = build_default_training_args()
    spec = resolve_training_run_spec(args)
    default_effective, default_max_steps, default_timesteps = _derive_training_budget(
        training_episodes=5000,
        num_envs=8,
        step_distance=30.0,
        rollout_steps_per_update=8192,
        schedule_time_s=465.0,
    )
    assert spec.training_episodes == 5000
    assert spec.max_episode_steps == default_max_steps
    assert spec.total_timesteps == default_timesteps
    budget = spec.run_metadata["training_budget"]
    assert budget["mode"] == "completed_episodes"
    assert budget["training_episodes"] == 5000
    assert budget["effective_training_episodes"] == default_effective
    assert budget["max_episode_steps"] == default_max_steps
    assert budget["derived_total_timesteps"] == default_timesteps

    # 5. 异常输入校验
    with pytest.raises(ValueError, match="training_episodes must be positive"):
        _ = _derive_training_budget(
            training_episodes=0,
            num_envs=8,
            step_distance=30.0,
            rollout_steps_per_update=8192,
            schedule_time_s=465.0,
        )


def test_train_single_experiment_persists_and_propagates_completed_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rl.experiment_utils import (
        evaluate_final_training_run,
        train_single_experiment,
    )

    args = build_default_training_args()
    args.training_episodes = 8
    args.output_root = str(tmp_path / "output")
    spec = resolve_training_run_spec(args)

    captured_learn_kwargs: dict[str, object] = {}

    class FakePPO:
        def __init__(self, *a: object, **kw: object) -> None:
            self.num_timesteps = 4096

        def learn(self, *a: object, callback: object = None, **kw: object) -> FakePPO:
            captured_learn_kwargs.update(kw)
            if hasattr(callback, "callbacks"):
                for cb in callback.callbacks:
                    if hasattr(cb, "_completed_episode_count"):
                        cb._completed_episode_count = 8
                    elif hasattr(cb, "completed_episode_count"):
                        cb.completed_episode_count = 8
                    if hasattr(cb, "n_episodes"):
                        cb.n_episodes = 8
            return self

        def save(self, *a: object, **kw: object) -> None:
            pass

        @classmethod
        def load(cls, *a: object, **kw: object) -> FakePPO:
            return cls()

    monkeypatch.setattr("rl.experiment_utils.PPO", FakePPO)

    trained_spec = train_single_experiment(args, spec=spec)

    with pytest.raises(FileExistsError, match="already contains training artifacts"):
        _ = train_single_experiment(args, spec=spec)

    # 1. Returned spec has completed budget metadata
    returned_budget = trained_spec.run_metadata.training_budget
    assert returned_budget is not None
    assert returned_budget.actual_completed_episodes == 8
    assert returned_budget.actual_training_timesteps == 4096
    assert returned_budget.target_reached is True
    assert returned_budget.stop_reason == "completed_episode_target"
    assert captured_learn_kwargs.get("progress_bar") is False

    # 2. Persisted metadata on disk has completed budget metadata
    persisted_meta = load_run_metadata(trained_spec.final_output_dir)
    persisted_budget = persisted_meta["training_budget"]
    assert persisted_budget["actual_completed_episodes"] == 8
    assert persisted_budget["actual_training_timesteps"] == 4096
    assert persisted_budget["target_reached"] is True
    assert persisted_budget["stop_reason"] == "completed_episode_target"

    # 3. Propagated spec into evaluate_final_training_run passes updated metadata
    captured_metadata: dict[str, object] = {}

    def fake_evaluate_and_save(
        model: object,
        env: object,
        output_path: str,
        metadata: dict[str, object],
        deterministic: bool = True,
        metrics_path: str | None = None,
    ) -> tuple[object, str, str]:
        captured_metadata.update(metadata)
        return (None, output_path, metrics_path or "")

    monkeypatch.setattr(
        "rl.experiment_utils.evaluate_and_save_final_policy",
        fake_evaluate_and_save,
    )

    _ = evaluate_final_training_run(trained_spec)
    final_budget = captured_metadata.get("training_budget")
    assert isinstance(final_budget, dict)
    assert final_budget["actual_completed_episodes"] == 8
    assert final_budget["actual_training_timesteps"] == 4096
    assert final_budget["target_reached"] is True
    assert final_budget["stop_reason"] == "completed_episode_target"


def test_train_single_experiment_uses_environment_step_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rl.callbacks import StopTrainingOnCompletedEpisodes
    from rl.experiment_utils import train_single_experiment

    args = build_default_training_args()
    args.budget_mode = "environment_steps"
    args.training_rollouts = 2
    args.output_root = str(tmp_path / "output")
    spec = resolve_training_run_spec(args)
    captured_callbacks: list[object] = []
    captured_learn_kwargs: dict[str, object] = {}

    class FakePPO:
        def __init__(self, *a: object, **kw: object) -> None:
            self.num_timesteps = 0

        def learn(
            self,
            *a: object,
            total_timesteps: int,
            callback: object = None,
            **kw: object,
        ) -> FakePPO:
            self.num_timesteps = total_timesteps
            captured_learn_kwargs.update(kw)
            if hasattr(callback, "callbacks"):
                captured_callbacks.extend(callback.callbacks)
                for item in callback.callbacks:
                    if hasattr(item, "_completed_episode_count"):
                        item._completed_episode_count = 3
            return self

        def save(self, *a: object, **kw: object) -> None:
            pass

    monkeypatch.setattr("rl.experiment_utils.PPO", FakePPO)

    trained = train_single_experiment(args, spec=spec)
    budget = trained.run_metadata.training_budget

    assert budget is not None
    assert budget.actual_training_timesteps == 16_384
    assert budget.actual_training_rollouts == 2
    assert budget.actual_completed_episodes == 3
    assert budget.target_reached is True
    assert budget.stop_reason == "environment_step_target"
    assert captured_learn_kwargs.get("progress_bar") is True
    assert not any(
        isinstance(item, StopTrainingOnCompletedEpisodes) for item in captured_callbacks
    )
