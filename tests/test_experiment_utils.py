import matplotlib.pyplot as plt
import pytest

from pathlib import Path

from rl.experiment_utils import (
    DEFAULT_REWARD_PROFILE_NAME,
    RUN_METADATA_FILENAME,
    add_panel_label,
    build_rl_trajectory_comparison_key,
    build_run_metadata,
    build_reward_config,
    load_run_metadata,
    resolve_output_dir,
    resolve_reward_profile,
    resolve_tb_log_name,
    reward_profile_names,
    save_run_metadata,
)


def test_reward_profile_names_include_real_pbrs_ablation_profiles() -> None:
    assert reward_profile_names() == (
        "basic",
        "basic_safety",
        "basic_safety_stopping",
    )


@pytest.mark.parametrize(
    "profile_name",
    ("full_shaping", "basic_safety_stopping_punctuality"),
)
def test_removed_punctuality_profiles_are_rejected(profile_name: str) -> None:
    with pytest.raises(ValueError):
        resolve_reward_profile(profile_name)


def test_reward_profiles_keep_energy_comfort_and_toggle_pbrs() -> None:
    expected_flags = {
        "basic": (False, False),
        "basic_safety": (True, False),
        "basic_safety_stopping": (True, True),
    }
    for profile_name, expected in expected_flags.items():
        reward_config = build_reward_config(profile_name)
        assert reward_config.enable_energy is True
        assert reward_config.enable_comfort is True
        assert (
            reward_config.enable_potential_safety,
            reward_config.enable_potential_stopping,
        ) == expected


def test_resolve_output_dir_scopes_non_default_profile_and_experiment_tag() -> None:
    output_dir = resolve_output_dir(
        output_root="output/optimal/rl",
        schedule_time_s=430.0,
        max_step_distance=100.0,
        reward_profile_name="basic",
        experiment_tag="Trial A",
    )

    assert Path(output_dir).name == "430p0_100p0__basic__trial_a"


def test_resolve_tb_log_name_generates_experiment_scoped_name() -> None:
    tb_log_name = resolve_tb_log_name(
        tb_log_name=None,
        run_mode="monitor_best",
        schedule_time_s=430.0,
        max_step_distance=100.0,
        reward_profile_name=DEFAULT_REWARD_PROFILE_NAME,
        experiment_tag=None,
    )

    assert tb_log_name == "train_log__monitor_best__430p0_100p0__basic_safety_stopping"


def test_load_run_metadata_falls_back_to_parent_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "430p0_100p0__basic"
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True)

    expected_metadata = {
        "reward_profile_name": "basic",
        "tb_log_name": "trainning_log__monitor_best__430p0_100p0__basic",
    }
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
        build_rl_trajectory_comparison_key(
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
