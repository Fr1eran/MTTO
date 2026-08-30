from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from rl.experiment_utils import (
    DEFAULT_COMFORT_REWARD_SCALE,
    DEFAULT_DEVICE,
    DEFAULT_ENERGY_REWARD_SCALE,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_PRESET_NAME,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    RUN_METADATA_FILENAME,
    add_panel_label,
    build_default_training_args,
    build_reward_config,
    build_rl_trajectory_comparison_key,
    build_run_metadata,
    curriculum_profile_names,
    load_run_metadata,
    resolve_curriculum_profile_name,
    resolve_output_dir,
    resolve_reward_preset,
    resolve_tb_log_name,
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


def test_curriculum_profiles_resolve_with_disabled_default() -> None:
    assert curriculum_profile_names() == ("none", "dspdl", "dspdl_completion")
    assert resolve_curriculum_profile_name() == "none"
    assert resolve_curriculum_profile_name("dspdl") == "dspdl"
    assert resolve_curriculum_profile_name("dspdl_completion") == "dspdl_completion"
    with pytest.raises(
        ValueError, match="Available profiles: none, dspdl, dspdl_completion"
    ):
        _ = resolve_curriculum_profile_name("fixed_reverse")


def test_curriculum_profile_scopes_nonbaseline_output_name() -> None:
    output_dir = resolve_output_dir(
        output_root="output/optimal/rl",
        schedule_time_s=430.0,
        step_distance=30.0,
        reward_preset_name="basic",
        curriculum_profile_name="dspdl",
    )

    assert Path(output_dir).name == "430p0_30p0__basic__dspdl"


def test_completion_curriculum_profile_has_an_independent_output_name() -> None:
    output_dir = resolve_output_dir(
        output_root="output/optimal/rl",
        schedule_time_s=430.0,
        step_distance=30.0,
        reward_preset_name="basic",
        curriculum_profile_name="dspdl_completion",
    )

    assert Path(output_dir).name == "430p0_30p0__basic__dspdl_completion"


def test_reward_preset_names_include_real_pbrs_ablation_profiles() -> None:
    assert reward_preset_names() == ("basic", "basic_safety")


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


def test_reward_presets_keep_energy_comfort_and_toggle_pbrs() -> None:
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


def test_resolve_output_dir_always_scopes_default_reward_preset() -> None:
    output_dir = resolve_output_dir(
        output_root="output/optimal/rl",
        schedule_time_s=430.0,
        step_distance=100.0,
        reward_preset_name=DEFAULT_REWARD_PRESET_NAME,
    )

    assert Path(output_dir).name == "430p0_100p0__basic_safety"


def test_resolve_output_dir_scopes_non_default_profile_and_experiment_tag() -> None:
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

    assert tb_log_name == "train_log__monitor_best__430p0_100p0__basic_safety"


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
    with pytest.raises(ValueError, match="required trajectory selection fields"):
        _ = build_rl_trajectory_comparison_key(
            {
                "success": True,
                "total_energy_j": 4_000.0,
                "stop_error_m": 0.2,
                "time_error_s": 1.0,
                "total_reward": 10.0,
            }
        )


def test_build_rl_trajectory_comparison_key_rebuilds_from_strict_limits() -> None:
    punctual_arrival = {
        "success": True,
        "total_energy_j": 8_000.0,
        "stop_error_m": 0.2,
        "time_error_s": 1.0,
        "total_reward": 10.0,
        "precise_arrival": True,
        "punctual_arrival": True,
        "strict_stop_error_limit_m": 0.3,
        "strict_time_error_limit_s": 5.0,
    }
    precise_arrival = {
        "success": True,
        "total_energy_j": 1_000.0,
        "stop_error_m": 0.2,
        "time_error_s": 8.0,
        "total_reward": 100.0,
        "precise_arrival": True,
        "punctual_arrival": False,
        "strict_stop_error_limit_m": 0.3,
        "strict_time_error_limit_s": 5.0,
    }

    assert build_rl_trajectory_comparison_key(
        punctual_arrival
    ) > build_rl_trajectory_comparison_key(precise_arrival)


def test_build_rl_trajectory_comparison_key_rebuilds_legacy_arrival_fields() -> None:
    punctual_arrival = {
        "success": True,
        "total_energy_j": 8_000.0,
        "stop_error_m": 0.2,
        "time_error_s": 1.0,
        "total_reward": 10.0,
        "strict_stop_error_limit_m": 0.3,
        "strict_time_error_limit_s": 5.0,
    }
    imprecise_arrival = {
        "success": True,
        "total_energy_j": 1_000.0,
        "stop_error_m": 0.5,
        "time_error_s": 1.0,
        "total_reward": 100.0,
        "strict_stop_error_limit_m": 0.3,
        "strict_time_error_limit_s": 5.0,
    }

    assert build_rl_trajectory_comparison_key(
        punctual_arrival
    ) > build_rl_trajectory_comparison_key(imprecise_arrival)


def test_build_rl_trajectory_comparison_key_treats_time_limit_as_exclusive() -> None:
    within_limit = {
        "success": True,
        "total_energy_j": 8_000.0,
        "stop_error_m": 0.2,
        "time_error_s": 4.9999,
        "total_reward": 10.0,
        "strict_stop_error_limit_m": 0.3,
        "strict_time_error_limit_s": 5.0,
    }
    exact_limit = {
        "success": True,
        "total_energy_j": 1_000.0,
        "stop_error_m": 0.2,
        "time_error_s": 5.0,
        "total_reward": 100.0,
        "strict_stop_error_limit_m": 0.3,
        "strict_time_error_limit_s": 5.0,
    }

    assert build_rl_trajectory_comparison_key(
        within_limit
    ) > build_rl_trajectory_comparison_key(exact_limit)


def test_add_panel_label_places_text_on_axes() -> None:
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
        _reward_config_to_dict,
        resolve_survival_reward_scale,
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

    d = _reward_config_to_dict(cfg_custom)
    assert d["energy_reward_scale"] == DEFAULT_ENERGY_REWARD_SCALE
    assert d["comfort_reward_scale"] == DEFAULT_COMFORT_REWARD_SCALE
    assert d["survival_reward_scale"] == 30.0
    assert d["enable_potential_safety"] is False
    assert "enable_energy" not in d
    assert "enable_comfort" not in d


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
    assert spec.training_episodes == 7000
    assert spec.max_episode_steps == max_steps
    assert spec.total_timesteps == derived_timesteps
    budget = spec.run_metadata["training_budget"]
    assert budget["mode"] == "completed_episodes"
    assert budget["training_episodes"] == 7000
    assert budget["effective_training_episodes"] == 7000
    assert budget["max_episode_steps"] == max_steps
    assert budget["derived_total_timesteps"] == derived_timesteps

    # 5. 异常输入校验
    with pytest.raises(ValueError, match="training_episodes must be positive"):
        _ = _derive_training_budget(
            training_episodes=0,
            num_envs=8,
            step_distance=30.0,
            rollout_steps_per_update=8192,
            schedule_time_s=465.0,
        )
