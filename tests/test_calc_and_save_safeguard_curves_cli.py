import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from model.vehicle import VehicleInfo
from scripts import calc_and_save_safeguard_curves as cli
from utils.data_loader import load_safeguard_curves


class FakeCalculator:
    def __init__(self) -> None:
        self.levi_kwargs = None
        self.brake_kwargs = None

    def calc_levi_and_min_curves(self, **kwargs):
        self.levi_kwargs = kwargs
        return [_curve(1.0)], [_curve(2.0)]

    def calc_brake_and_max_curves(self, **kwargs):
        self.brake_kwargs = kwargs
        return [_curve(3.0)], [_curve(4.0)]


def _curve(speed: float) -> np.ndarray:
    return np.asarray([[0.0, 1.0], [speed, 0.0]], dtype=np.float64)


def _inputs(calculator=None) -> cli.CalculationInputs:
    return cli.CalculationInputs(
        calculator=calculator or FakeCalculator(),
        vehicle=VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5),
        accessible_points=np.asarray([100.0], dtype=np.float64),
        dangerous_points=np.asarray([200.0], dtype=np.float64),
    )


def _curves() -> dict[str, list[np.ndarray]]:
    return {
        "levi_curves_list": [_curve(1.0)],
        "brake_curves_list": [_curve(2.0)],
        "min_curves_list": [_curve(3.0)],
        "max_curves_list": [_curve(4.0)],
    }


def test_cli_defaults_match_previous_calculation() -> None:
    args = cli.build_cli_parser().parse_args([])

    assert args.output_dir == Path("output/safeguardcurves")
    assert args.distance_step_m == pytest.approx(1.0)
    assert args.mass_tonnes == pytest.approx(317.5)
    assert args.trainset_count == 5
    assert args.max_acceleration_mps2 == pytest.approx(1.0)
    assert args.max_deceleration_mps2 == pytest.approx(1.0)
    assert args.position_error_m == pytest.approx(1.0)
    assert args.speed_error_mps == pytest.approx(0.1)
    assert args.traction_cutoff_delay_s == pytest.approx(0.5)
    assert args.vortex_brake_delay_s == pytest.approx(0.5)
    assert args.min_curve_position_offset_m == pytest.approx(0.0)
    assert args.include_acceleration_zone_end is True
    assert args.force is False
    assert args.dry_run is False


def test_cli_accepts_explicit_calculation_parameters() -> None:
    args = cli.build_cli_parser().parse_args(
        [
            "--output-dir",
            "output/custom",
            "--distance-step-m",
            "0.5",
            "--mass-tonnes",
            "300",
            "--trainset-count",
            "4",
            "--max-acceleration-mps2",
            "0.9",
            "--max-deceleration-mps2",
            "1.1",
            "--position-error-m",
            "2",
            "--speed-error-mps",
            "0.2",
            "--traction-cutoff-delay-s",
            "0.6",
            "--vortex-brake-delay-s",
            "0.7",
            "--min-curve-position-offset-m",
            "-3",
            "--no-include-acceleration-zone-end",
            "--force",
            "--dry-run",
        ]
    )

    config = cli._config_from_args(args)
    assert config.distance_step_m == pytest.approx(0.5)
    assert config.mass_tonnes == pytest.approx(300.0)
    assert config.trainset_count == 4
    assert config.max_acceleration_mps2 == pytest.approx(0.9)
    assert config.max_deceleration_mps2 == pytest.approx(1.1)
    assert config.position_error_m == pytest.approx(2.0)
    assert config.speed_error_mps == pytest.approx(0.2)
    assert config.traction_cutoff_delay_s == pytest.approx(0.6)
    assert config.vortex_brake_delay_s == pytest.approx(0.7)
    assert config.min_curve_position_offset_m == pytest.approx(-3.0)
    assert config.include_acceleration_zone_end is False
    assert args.force is True
    assert args.dry_run is True


@pytest.mark.parametrize(
    "option,value",
    [
        ("--distance-step-m", "0"),
        ("--mass-tonnes", "nan"),
        ("--trainset-count", "0"),
        ("--max-acceleration-mps2", "-1"),
        ("--max-deceleration-mps2", "0"),
        ("--position-error-m", "-1"),
        ("--speed-error-mps", "-0.1"),
        ("--traction-cutoff-delay-s", "inf"),
        ("--vortex-brake-delay-s", "-1"),
        ("--min-curve-position-offset-m", "nan"),
    ],
)
def test_main_rejects_invalid_parameters(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([option, value])

    assert exc_info.value.code == 2


def test_build_inputs_adds_acceleration_zone_before_track_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "load_slopes",
        lambda: (np.asarray([0.0]), np.asarray([0.0, 1000.0])),
    )
    monkeypatch.setattr(
        cli,
        "load_speed_limits",
        lambda to_mps: (np.asarray([100.0]), np.asarray([0.0, 1000.0])),
    )
    monkeypatch.setattr(
        cli,
        "load_auxiliary_stopping_areas_ap_and_dp",
        lambda: ([100.0], [200.0]),
    )
    monkeypatch.setattr(
        cli,
        "load_acceleration_zones",
        lambda: {"uplink": {"start": 10.0, "end": 50.0}},
    )

    inputs = cli.build_calculation_inputs(cli.SafeguardCurveConfig())

    np.testing.assert_array_equal(inputs.dangerous_points, [50.0, 200.0])
    assert inputs.calculator.track.ASA_dps == [50.0, 200.0]
    assert inputs.vehicle.max_dec == pytest.approx(-1.0)

    without_zone = cli.build_calculation_inputs(
        cli.SafeguardCurveConfig(include_acceleration_zone_end=False)
    )
    np.testing.assert_array_equal(without_zone.dangerous_points, [200.0])
    assert without_zone.calculator.track.ASA_dps == [200.0]


def test_calculate_curves_translates_error_signs_and_parameters() -> None:
    calculator = FakeCalculator()
    inputs = _inputs(calculator)
    config = cli.SafeguardCurveConfig(
        distance_step_m=0.5,
        position_error_m=2.0,
        speed_error_mps=0.2,
        traction_cutoff_delay_s=0.6,
        vortex_brake_delay_s=0.7,
        min_curve_position_offset_m=-3.0,
    )

    curves = cli.calculate_curves(config, inputs)

    assert set(curves) == set(cli.CURVE_FILENAMES)
    assert calculator.levi_kwargs["ds"] == pytest.approx(0.5)
    assert calculator.levi_kwargs["pos_error"] == pytest.approx(2.0)
    assert calculator.levi_kwargs["speed_error"] == pytest.approx(0.2)
    assert calculator.levi_kwargs["pos_offset"] == pytest.approx(-3.0)
    assert calculator.levi_kwargs["delay_time_until_DPS_done"] == pytest.approx(0.6)
    assert calculator.brake_kwargs["ds"] == pytest.approx(0.5)
    assert calculator.brake_kwargs["pos_error"] == pytest.approx(-2.0)
    assert calculator.brake_kwargs["speed_error"] == pytest.approx(-0.2)
    assert calculator.brake_kwargs["delay_time_until_DPS_done"] == pytest.approx(0.6)
    assert calculator.brake_kwargs["delay_time_until_VB_begin"] == pytest.approx(0.7)


def test_dry_run_does_not_create_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "nested" / "curves"
    monkeypatch.setattr(cli, "build_calculation_inputs", lambda config: _inputs())
    monkeypatch.setattr(
        cli,
        "calculate_curves",
        lambda *args: pytest.fail("dry-run must not calculate curves"),
    )

    assert cli.main(["--output-dir", str(output_dir), "--dry-run"]) == 0

    assert not output_dir.exists()
    assert str(output_dir) in capsys.readouterr().out


def test_existing_artifact_requires_force_before_calculation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "curves"
    output_dir.mkdir()
    (output_dir / "levi_curves_list.pkl").write_bytes(b"old")
    monkeypatch.setattr(cli, "build_calculation_inputs", lambda config: _inputs())
    monkeypatch.setattr(
        cli,
        "calculate_curves",
        lambda *args: pytest.fail("calculation must not start without --force"),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--output-dir", str(output_dir)])

    assert exc_info.value.code == 2
    assert (output_dir / "levi_curves_list.pkl").read_bytes() == b"old"


def test_main_saves_curves_metadata_and_custom_loader_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "curves"
    inputs = _inputs()
    curves = _curves()
    monkeypatch.setattr(cli, "build_calculation_inputs", lambda config: inputs)
    monkeypatch.setattr(cli, "calculate_curves", lambda config, actual: curves)

    assert cli.main(["--output-dir", str(output_dir)]) == 0

    loaded = load_safeguard_curves(
        "levi_curves_list",
        "max_curves_list",
        curve_dir=output_dir,
    )
    np.testing.assert_array_equal(loaded[0][0], curves["levi_curves_list"][0])
    np.testing.assert_array_equal(loaded[1][0], curves["max_curves_list"][0])

    metadata = json.loads((output_dir / "metadata.json").read_text("utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["vehicle"] == {"length_m": 128.5}
    assert metadata["scenario"]["accessible_points_m"] == [100.0]
    assert metadata["artifacts"]["max_curves_list"] == {
        "filename": "max_curves_list.pkl",
        "curve_count": 1,
    }
    assert not list(output_dir.glob("*.tmp"))


def test_save_serialization_failure_preserves_existing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "curves"
    output_dir.mkdir()
    existing = output_dir / "levi_curves_list.pkl"
    existing.write_bytes(b"old")

    def fail_dump(*args, **kwargs) -> None:
        raise OSError("serialization failed")

    monkeypatch.setattr(cli.pickle, "dump", fail_dump)

    with pytest.raises(OSError, match="serialization failed"):
        cli.save_curves(
            curves=_curves(),
            metadata={"schema_version": 1},
            output_dir=output_dir,
        )

    assert existing.read_bytes() == b"old"
    assert not list(output_dir.glob("*.tmp"))


def test_default_loader_behavior_remains_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [_curve(8.0)]
    curve_path = tmp_path / "output" / "safeguardcurves"
    curve_path.mkdir(parents=True)
    with (curve_path / "sample.pkl").open("wb") as file:
        pickle.dump(expected, file)
    monkeypatch.setattr("utils.data_loader.PROJECT_ROOT", tmp_path)

    (loaded,) = load_safeguard_curves("sample")

    np.testing.assert_array_equal(loaded[0], expected[0])
