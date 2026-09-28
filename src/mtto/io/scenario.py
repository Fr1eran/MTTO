from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np

from mtto.domain.dynamics import Vehicle
from mtto.domain.energy import EnergyParams
from mtto.domain.line import Line
from mtto.domain.safeguard import (
    CalculationInputs,
    Safeguard,
    SafeguardCurveConfig,
    SafeGuardCurves,
    SafeguardParams,
    build_safeguard,
    calculate_curves,
)
from mtto.domain.scenario import (
    Scenario,
    ScheduleChange,
    Task,
)


def compute_scenario_hash(
    *,
    line: Line,
    vehicle: Vehicle,
    energy: EnergyParams,
    safeguard: Safeguard,
    prepend_acceleration_zone_end: bool,
) -> str:
    """Calculate deterministic SHA-256 hash for scenario parameters."""
    params = safeguard.params
    payload = {
        "acceleration_zone_end_prepended": prepend_acceleration_zone_end,
        "accessible_points": list(line.accessible_points_m),
        "dangerous_points": list(line.danger_points_m),
        "energy": {
            "L_d": energy.L_d,
            "L_k": energy.L_k,
            "Phi_1": energy.Phi_1,
            "Phi_2": energy.Phi_2,
            "Psi_fd": energy.Psi_fd,
            "R_k": energy.R_k,
            "R_m": energy.R_m,
            "Tau": energy.Tau,
            "k_c": energy.k_c,
        },
        "line": {
            "slope_intervals": line.slope_intervals.tolist(),
            "slopes": line.slopes.tolist(),
            "speed_limit_intervals": line.speed_limit_intervals.tolist(),
            "speed_limits": line.speed_limits.tolist(),
        },
        "safeguard_params": {
            "distance_step_m": params.distance_step_m,
            "factor": params.factor,
            "generation_danger_points_m": list(params.generation_danger_points_m),
            "min_curve_position_offset_m": params.min_curve_position_offset_m,
            "position_error_m": params.position_error_m,
            "speed_error_mps": params.speed_error_mps,
            "step_delay_s": params.step_delay_s,
            "traction_cutoff_delay_s": params.traction_cutoff_delay_s,
            "vortex_brake_delay_s": params.vortex_brake_delay_s,
        },
        "vehicle": {
            "length": vehicle.length,
            "levi_power_per_mass": vehicle.levi_power_per_mass,
            "mass": vehicle.mass,
            "max_acc": vehicle.max_acc,
            "max_dec": vehicle.max_dec,
            "max_slope_capacity": vehicle.max_slope_capacity,
            "max_speed": vehicle.max_speed,
            "numoftrainsets": vehicle.numoftrainsets,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_tasks(spec_path: Path) -> dict[str, Task]:
    """Load tasks from a TOML specification file."""
    with spec_path.open("rb") as f:
        data = tomllib.load(f)

    tasks: dict[str, Task] = {}
    for task_name, task_dict in data.items():
        schedule_change = None
        sc_dict = task_dict.get("schedule_change")
        if sc_dict:
            schedule_change = ScheduleChange(
                trigger_position_m=float(sc_dict["trigger_position_m"]),
                new_schedule_time_s=float(sc_dict["new_schedule_time_s"]),
            )

        schedule_time_s = task_dict.get("schedule_time_s")
        if schedule_time_s is not None:
            schedule_time_s = float(schedule_time_s)

        tasks[task_name] = Task(
            start_position_m=float(task_dict["start_position_m"]),
            target_position_m=float(task_dict["target_position_m"]),
            schedule_time_s=schedule_time_s,
            max_acc_change=float(task_dict["max_acc_change"]),
            max_stop_error_m=float(task_dict["max_stop_error_m"]),
            max_arr_time_error_s=float(task_dict["max_arr_time_error_s"]),
            schedule_change=schedule_change,
        )
    return tasks


def load_scenario(spec_path: Path, line_dir: Path) -> Scenario:
    """Load scenario from TOML specification and line JSON directory."""
    with spec_path.open("rb") as f:
        toml_data = tomllib.load(f)

    vehicle_cfg = toml_data["vehicle"]
    energy_cfg = toml_data["energy"]
    safeguard_cfg = toml_data["safeguard"]

    vehicle = Vehicle(
        mass=float(vehicle_cfg["mass"]),
        numoftrainsets=int(vehicle_cfg["numoftrainsets"]),
        length=float(vehicle_cfg["length"]),
        max_speed=float(vehicle_cfg["max_speed"]),
        max_acc=float(vehicle_cfg["max_acc"]),
        max_dec=float(vehicle_cfg["max_dec"]),
        max_slope_capacity=float(vehicle_cfg["max_slope_capacity"]),
        levi_power_per_mass=float(vehicle_cfg["levi_power_per_mass"]),
    )

    energy = EnergyParams(
        R_m=float(energy_cfg["R_m"]),
        L_d=float(energy_cfg["L_d"]),
        R_k=float(energy_cfg["R_k"]),
        L_k=float(energy_cfg["L_k"]),
        Tau=float(energy_cfg["Tau"]),
        Psi_fd=float(energy_cfg["Psi_fd"]),
        k_c=float(energy_cfg["k_c"]),
        Phi_1=float(energy_cfg["Phi_1"]),
        Phi_2=float(energy_cfg["Phi_2"]),
    )

    with (line_dir / "slopes.json").open("r", encoding="utf-8") as f:
        slopes_data = json.load(f)
    with (line_dir / "speed_limits.json").open("r", encoding="utf-8") as f:
        speed_limits_data = json.load(f)
    with (line_dir / "auxiliary_parking_areas.json").open("r", encoding="utf-8") as f:
        apa_data = json.load(f)
    with (line_dir / "acceleration_zones.json").open("r", encoding="utf-8") as f:
        accel_data = json.load(f)

    slopes = np.asarray(slopes_data["slopes"], dtype=np.float64)
    slope_intervals = np.asarray(slopes_data["intervals"], dtype=np.float64)
    speed_limits = np.asarray(speed_limits_data["speed_limits"], dtype=np.float64) / 3.6
    speed_limit_intervals = np.asarray(speed_limits_data["intervals"], dtype=np.float64)

    accessible_points = [float(p) for p in apa_data["accessible_points"]]
    line_dangerous_points = [float(p) for p in apa_data["dangerous_points"]]

    prepend_accel = bool(safeguard_cfg["prepend_acceleration_zone_end"])
    if prepend_accel:
        accel_end = float(accel_data["uplink"]["end"])
        generation_danger_points = (accel_end, *line_dangerous_points)
    else:
        generation_danger_points = tuple(line_dangerous_points)

    params = SafeguardParams(
        factor=float(safeguard_cfg["factor"]),
        step_delay_s=float(safeguard_cfg["step_delay_s"]),
        distance_step_m=float(safeguard_cfg["distance_step_m"]),
        position_error_m=float(safeguard_cfg["position_error_m"]),
        speed_error_mps=float(safeguard_cfg["speed_error_mps"]),
        traction_cutoff_delay_s=float(safeguard_cfg["traction_cutoff_delay_s"]),
        vortex_brake_delay_s=float(safeguard_cfg["vortex_brake_delay_s"]),
        min_curve_position_offset_m=float(safeguard_cfg["min_curve_position_offset_m"]),
        generation_danger_points_m=generation_danger_points,
    )

    curve_config = SafeguardCurveConfig(
        distance_step_m=params.distance_step_m,
        mass_tonnes=vehicle.mass,
        trainset_count=vehicle.numoftrainsets,
        max_acceleration_mps2=vehicle.max_acc,
        max_deceleration_mps2=vehicle.max_dec_abs,
        position_error_m=params.position_error_m,
        speed_error_mps=params.speed_error_mps,
        traction_cutoff_delay_s=params.traction_cutoff_delay_s,
        vortex_brake_delay_s=params.vortex_brake_delay_s,
        min_curve_position_offset_m=params.min_curve_position_offset_m,
    )

    # Normalization: ensure all arrays are float64 and read-only
    line_slopes = np.array(slopes, dtype=np.float64, copy=True)
    line_slopes.flags.writeable = False
    line_slope_intervals = np.array(slope_intervals, dtype=np.float64, copy=True)
    line_slope_intervals.flags.writeable = False
    line_speed_limits = np.array(speed_limits, dtype=np.float64, copy=True)
    line_speed_limits.flags.writeable = False
    line_speed_limit_intervals = np.array(
        speed_limit_intervals, dtype=np.float64, copy=True
    )
    line_speed_limit_intervals.flags.writeable = False

    line = Line(
        slopes=line_slopes,
        slope_intervals=line_slope_intervals,
        speed_limits=line_speed_limits,
        speed_limit_intervals=line_speed_limit_intervals,
        accessible_points_m=tuple(accessible_points),
        danger_points_m=tuple(line_dangerous_points),
    )

    line_for_curves = Line(
        slopes=line_slopes,
        slope_intervals=line_slope_intervals,
        speed_limits=line_speed_limits,
        speed_limit_intervals=line_speed_limit_intervals,
        accessible_points_m=tuple(accessible_points),
        danger_points_m=tuple(generation_danger_points),
    )

    calculation_inputs = CalculationInputs(
        calculator=SafeGuardCurves(track=line_for_curves),
        vehicle=vehicle,
        accessible_points=np.asarray(accessible_points, dtype=np.float64),
        dangerous_points=np.asarray(generation_danger_points, dtype=np.float64),
    )

    curves = calculate_curves(curve_config, calculation_inputs)

    safeguard = build_safeguard(
        params=params,
        line=line,
        levi_curves=curves["levi_curves_list"],
        brake_curves=curves["brake_curves_list"],
        min_curves=curves["min_curves_list"],
        max_curves=curves["max_curves_list"],
    )

    scenario_hash = compute_scenario_hash(
        line=line,
        vehicle=vehicle,
        energy=energy,
        safeguard=safeguard,
        prepend_acceleration_zone_end=prepend_accel,
    )

    return Scenario(
        line=line,
        vehicle=vehicle,
        energy=energy,
        safeguard=safeguard,
        scenario_hash=scenario_hash,
    )
