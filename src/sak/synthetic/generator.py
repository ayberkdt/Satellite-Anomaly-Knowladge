"""Mission-like synthetic satellite telemetry and partition-aware injections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

TELEMETRY_CHANNELS = (
    "battery_voltage",
    "battery_current",
    "battery_temperature",
    "battery_state_of_charge",
    "bus_voltage",
    "bus_current",
    "solar_array_voltage",
    "solar_array_current",
    "panel_temperature",
    "payload_temperature",
    "radiator_temperature",
    "heater_state",
    "reaction_wheel_speed_x",
    "reaction_wheel_speed_y",
    "reaction_wheel_speed_z",
    "reaction_wheel_current_x",
    "reaction_wheel_current_y",
    "reaction_wheel_current_z",
    "gyro_rate_x",
    "gyro_rate_y",
    "gyro_rate_z",
    "attitude_error",
    "transmitter_power",
    "receiver_rssi",
    "receiver_temperature",
    "antenna_temperature",
    "link_margin",
    "payload_current",
    "payload_mode",
)

CHANNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "EPS": (
        "battery_voltage",
        "battery_current",
        "bus_voltage",
        "bus_current",
        "solar_array_current",
        "solar_array_voltage",
        "battery_temperature",
        "battery_state_of_charge",
    ),
    "THERMAL": (
        "panel_temperature",
        "payload_temperature",
        "battery_temperature",
        "radiator_temperature",
        "heater_state",
    ),
    "AOCS": (
        "reaction_wheel_speed_x",
        "reaction_wheel_speed_y",
        "reaction_wheel_speed_z",
        "reaction_wheel_current_x",
        "reaction_wheel_current_y",
        "reaction_wheel_current_z",
        "gyro_rate_x",
        "gyro_rate_y",
        "gyro_rate_z",
        "attitude_error",
    ),
    "COMM": (
        "transmitter_power",
        "receiver_rssi",
        "receiver_temperature",
        "antenna_temperature",
        "link_margin",
    ),
    "PAYLOAD": (
        "payload_current",
        "payload_temperature",
        "payload_mode",
    ),
}

ANOMALY_TYPES = (
    "reaction_wheel_friction_increase",
    "battery_degradation",
    "solar_array_underperformance",
    "heater_stuck_on",
    "heater_stuck_off",
    "payload_overcurrent",
    "communication_link_margin_drop",
    "sensor_bias_drift",
    "battery_voltage_sag",
    "thermal_runaway",
)

BENIGN_TYPES = (
    "mode_transition_transient",
    "eclipse_thermal_transient",
    "safe_mode_transition",
)

DEFAULT_ANOMALY_SCHEDULE: dict[str, dict[str, Any]] = {
    "calibration": {
        "enabled": True,
        "count_per_type": 1,
        "severity_levels": ["medium"],
    },
    "validation": {
        "enabled": True,
        "count_per_type": 1,
        "severity_levels": ["low", "medium"],
    },
    "test": {
        "enabled": True,
        "count_per_type": 2,
        "severity_levels": ["low", "medium", "high"],
    },
}

SEVERITY_SCALE = {"low": 0.65, "medium": 1.0, "high": 1.45}


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for mission-like telemetry and anomaly scheduling."""

    periods: int = 14 * 24 * 60
    frequency: str = "1min"
    start: str = "2026-01-01T00:00:00Z"
    orbit_period_steps: int = 96
    seed: int = 42
    missing_fraction: float = 0.001
    train_fraction: float = 0.50
    calibration_fraction: float = 0.20
    validation_fraction: float = 0.10
    anomaly_schedule: dict[str, dict[str, Any]] | None = None
    benign_events_per_partition: int = 2


@dataclass(frozen=True)
class InjectionRecord:
    """Ground-truth metadata for one partition-scoped operational event."""

    event_id: str
    partition: str
    anomaly_type: str
    severity: str
    start: pd.Timestamp
    end: pd.Timestamp
    affected_channels: tuple[str, ...]
    expected_subsystem: str
    early_warning_region_start: pd.Timestamp
    failure_region_start: pd.Timestamp
    event_class: str
    notes: str
    parameters: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "start",
            "end",
            "early_warning_region_start",
            "failure_region_start",
        ):
            payload[key] = getattr(self, key).isoformat()
        payload["affected_channels"] = list(self.affected_channels)
        return payload


@dataclass(frozen=True)
class SyntheticDataset:
    """Generated telemetry, event manifest and canonical channel metadata."""

    frame: pd.DataFrame
    events: tuple[InjectionRecord, ...]
    channel_names: tuple[str, ...] = TELEMETRY_CHANNELS
    context_columns: tuple[str, ...] = (
        "orbit_phase",
        "eclipse",
        "sunlight",
        "beta_angle",
        "operational_mode",
        "maneuver_flag",
        "safe_mode_flag",
    )
    channel_groups: dict[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self.channel_groups is None:
            object.__setattr__(self, "channel_groups", CHANNEL_GROUPS)


def _ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _nominal_frame(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    start = pd.Timestamp(config.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    timestamps = pd.date_range(start=start, periods=config.periods, freq=config.frequency)
    step = np.arange(config.periods)
    orbit_phase = (step % config.orbit_period_steps) / config.orbit_period_steps
    sunlight = orbit_phase < 0.62
    eclipse = ~sunlight
    beta_angle = 18.0 * np.sin(2.0 * np.pi * step / (7 * 24 * 60))

    payload_cycle = (step // 240) % 3
    operational_mode = np.where(payload_cycle == 1, "payload", "nominal").astype(object)
    maneuver_flag = (step % (36 * 60)) < 18
    safe_mode_flag = (step % (5 * 24 * 60)) < 22
    operational_mode[maneuver_flag] = "maneuver"
    operational_mode[safe_mode_flag] = "safe"
    payload_active = operational_mode == "payload"
    payload_mode = payload_active.astype(float)

    orbit_wave = np.sin(2.0 * np.pi * orbit_phase)
    solar_array_current = np.clip(
        sunlight * (4.8 + 0.38 * orbit_wave + 0.004 * beta_angle)
        + rng.normal(0.0, 0.07, config.periods),
        0.0,
        None,
    )
    solar_array_voltage = (
        sunlight * (31.2 + 0.45 * orbit_wave)
        + eclipse * 0.6
        + rng.normal(0.0, 0.08, config.periods)
    )
    payload_current = (
        0.18
        + 1.75 * payload_active
        + 0.10 * maneuver_flag
        + rng.normal(0.0, 0.04, config.periods)
    )
    bus_current = (
        1.10
        + payload_current
        + 0.35 * maneuver_flag
        - 0.12 * safe_mode_flag
        + rng.normal(0.0, 0.04, config.periods)
    )
    battery_current = bus_current - 0.72 * solar_array_current

    state_of_charge = np.empty(config.periods, dtype=float)
    state_of_charge[0] = 78.0
    for index in range(1, config.periods):
        state_of_charge[index] = np.clip(
            state_of_charge[index - 1] - battery_current[index] * 0.010,
            48.0,
            96.0,
        )

    heater_state = ((eclipse) & (orbit_phase > 0.72)).astype(float)
    thermal_driver = (
        0.75 * sunlight
        + 0.40 * payload_active
        + 0.25 * heater_state
    )
    thermal_state = _ewma(thermal_driver.astype(float), alpha=0.035)
    panel_temperature = (
        4.0
        + 29.0 * thermal_state
        + 2.2 * orbit_wave
        + rng.normal(0.0, 0.28, config.periods)
    )
    radiator_temperature = (
        -3.0
        + 0.55 * panel_temperature
        - 5.5 * eclipse
        + rng.normal(0.0, 0.22, config.periods)
    )
    payload_temperature = (
        15.0
        + 8.5 * _ewma(payload_active.astype(float), alpha=0.025)
        + 0.15 * panel_temperature
        + rng.normal(0.0, 0.20, config.periods)
    )
    battery_temperature = (
        18.0
        + 0.16 * panel_temperature
        + 0.55 * np.maximum(battery_current, 0.0)
        + rng.normal(0.0, 0.16, config.periods)
    )
    battery_voltage = (
        27.2
        + 0.014 * state_of_charge
        - 0.20 * np.maximum(battery_current, 0.0)
        + rng.normal(0.0, 0.035, config.periods)
    )
    bus_voltage = (
        28.1
        - 0.052 * np.maximum(bus_current, 0.0)
        + 0.035 * sunlight
        + rng.normal(0.0, 0.025, config.periods)
    )

    maneuver_wave = maneuver_flag * np.sin(0.15 * step)
    gyro_rate_x = 0.015 * np.sin(4.0 * np.pi * orbit_phase) + 0.08 * maneuver_wave
    gyro_rate_y = 0.012 * np.cos(4.0 * np.pi * orbit_phase) - 0.06 * maneuver_wave
    gyro_rate_z = 0.010 * np.sin(6.0 * np.pi * orbit_phase) + 0.05 * maneuver_wave
    gyro_rate_x += rng.normal(0.0, 0.004, config.periods)
    gyro_rate_y += rng.normal(0.0, 0.004, config.periods)
    gyro_rate_z += rng.normal(0.0, 0.004, config.periods)
    attitude_error = (
        0.05
        + 0.9 * np.sqrt(gyro_rate_x**2 + gyro_rate_y**2 + gyro_rate_z**2)
        + rng.normal(0.0, 0.006, config.periods)
    )
    wheel_base = 1850.0 + 95.0 * np.sin(2.0 * np.pi * orbit_phase + 0.5)
    reaction_wheel_speed_x = wheel_base + 180.0 * maneuver_wave
    reaction_wheel_speed_y = 0.92 * wheel_base - 150.0 * maneuver_wave
    reaction_wheel_speed_z = 1.08 * wheel_base + 120.0 * maneuver_wave
    reaction_wheel_speed_x += rng.normal(0.0, 8.0, config.periods)
    reaction_wheel_speed_y += rng.normal(0.0, 8.0, config.periods)
    reaction_wheel_speed_z += rng.normal(0.0, 8.0, config.periods)
    reaction_wheel_current_x = 0.18 + 0.00016 * abs(reaction_wheel_speed_x)
    reaction_wheel_current_y = 0.18 + 0.00016 * abs(reaction_wheel_speed_y)
    reaction_wheel_current_z = 0.18 + 0.00016 * abs(reaction_wheel_speed_z)
    reaction_wheel_current_x += rng.normal(0.0, 0.012, config.periods)
    reaction_wheel_current_y += rng.normal(0.0, 0.012, config.periods)
    reaction_wheel_current_z += rng.normal(0.0, 0.012, config.periods)

    transmitter_power = (
        6.0
        + 2.4 * payload_active
        - 1.5 * safe_mode_flag
        + rng.normal(0.0, 0.08, config.periods)
    )
    antenna_temperature = (
        5.0 + 0.38 * panel_temperature + rng.normal(0.0, 0.18, config.periods)
    )
    receiver_temperature = (
        12.0 + 0.32 * antenna_temperature + rng.normal(0.0, 0.15, config.periods)
    )
    receiver_rssi = (
        -72.0
        + 2.8 * np.cos(2.0 * np.pi * orbit_phase)
        + 0.25 * transmitter_power
        + rng.normal(0.0, 0.35, config.periods)
    )
    link_margin = (
        12.0
        + 0.55 * (receiver_rssi + 72.0)
        + 0.45 * transmitter_power
        - 0.05 * antenna_temperature
        + rng.normal(0.0, 0.22, config.periods)
    )

    return pd.DataFrame(
        {
            "battery_voltage": battery_voltage,
            "battery_current": battery_current,
            "battery_temperature": battery_temperature,
            "battery_state_of_charge": state_of_charge,
            "bus_voltage": bus_voltage,
            "bus_current": bus_current,
            "solar_array_voltage": solar_array_voltage,
            "solar_array_current": solar_array_current,
            "panel_temperature": panel_temperature,
            "payload_temperature": payload_temperature,
            "radiator_temperature": radiator_temperature,
            "heater_state": heater_state,
            "reaction_wheel_speed_x": reaction_wheel_speed_x,
            "reaction_wheel_speed_y": reaction_wheel_speed_y,
            "reaction_wheel_speed_z": reaction_wheel_speed_z,
            "reaction_wheel_current_x": reaction_wheel_current_x,
            "reaction_wheel_current_y": reaction_wheel_current_y,
            "reaction_wheel_current_z": reaction_wheel_current_z,
            "gyro_rate_x": gyro_rate_x,
            "gyro_rate_y": gyro_rate_y,
            "gyro_rate_z": gyro_rate_z,
            "attitude_error": attitude_error,
            "transmitter_power": transmitter_power,
            "receiver_rssi": receiver_rssi,
            "receiver_temperature": receiver_temperature,
            "antenna_temperature": antenna_temperature,
            "link_margin": link_margin,
            "payload_current": payload_current,
            "payload_mode": payload_mode,
            "operational_mode": operational_mode,
            "eclipse": eclipse,
            "sunlight": sunlight,
            "orbit_phase": orbit_phase,
            "beta_angle": beta_angle,
            "maneuver_flag": maneuver_flag,
            "safe_mode_flag": safe_mode_flag,
            "is_anomaly": False,
            "anomaly_event_id": "",
            "anomaly_type": "",
            "label_taxonomy": "nominal",
        },
        index=timestamps,
    ).rename_axis("timestamp")


def _partition_bounds(config: SyntheticConfig) -> dict[str, tuple[int, int]]:
    fractions = (
        config.train_fraction,
        config.calibration_fraction,
        config.validation_fraction,
    )
    if any(value <= 0.0 for value in fractions) or sum(fractions) >= 1.0:
        raise ValueError("split fractions must be positive and leave a test partition")
    train_end = int(config.periods * config.train_fraction)
    calibration_end = train_end + int(config.periods * config.calibration_fraction)
    validation_end = calibration_end + int(config.periods * config.validation_fraction)
    return {
        "train": (0, train_end),
        "calibration": (train_end, calibration_end),
        "validation": (calibration_end, validation_end),
        "test": (validation_end, config.periods),
    }


def _event_regions(
    early_start: int,
    severity: str,
    partition_end: int,
) -> tuple[int, int, int, int]:
    severity_index = {"low": 0, "medium": 1, "high": 2}[severity]
    precursor_steps = 12 + 4 * severity_index
    anomaly_steps = 48 + 12 * severity_index
    onset = early_start + precursor_steps
    end = min(onset + anomaly_steps, partition_end)
    critical = onset + max(1, int((end - onset) * 0.70))
    return early_start, onset, critical, end


def _set_values(
    frame: pd.DataFrame,
    indices: np.ndarray,
    channel: str,
    values: np.ndarray | float,
    *,
    operation: str = "add",
) -> None:
    location = frame.columns.get_loc(channel)
    if operation == "add":
        frame.iloc[indices, location] += values
    elif operation == "set":
        frame.iloc[indices, location] = values
    else:
        raise ValueError(f"unsupported operation: {operation}")


def _inject_anomaly(
    frame: pd.DataFrame,
    *,
    anomaly_type: str,
    partition: str,
    severity: str,
    early_start: int,
    partition_end: int,
    rng: np.random.Generator,
    event_number: int,
) -> InjectionRecord:
    early_start, onset, critical, end = _event_regions(
        early_start,
        severity,
        partition_end,
    )
    indices = np.arange(early_start, end)
    progress = np.linspace(0.05, 1.0, len(indices), endpoint=True)
    scale = SEVERITY_SCALE[severity]
    parameters: dict[str, float] = {"severity_scale": scale}

    if anomaly_type == "reaction_wheel_friction_increase":
        channels = (
            "reaction_wheel_current_x",
            "reaction_wheel_speed_x",
            "attitude_error",
        )
        subsystem = "AOCS"
        _set_values(frame, indices, "reaction_wheel_current_x", 0.35 * scale * progress)
        _set_values(frame, indices, "reaction_wheel_speed_x", -120.0 * scale * progress)
        _set_values(frame, indices, "attitude_error", 0.12 * scale * progress)
    elif anomaly_type == "battery_degradation":
        channels = (
            "battery_state_of_charge",
            "battery_voltage",
            "battery_temperature",
        )
        subsystem = "EPS"
        _set_values(frame, indices, "battery_state_of_charge", -7.0 * scale * progress**2)
        _set_values(frame, indices, "battery_voltage", -0.7 * scale * progress**2)
        _set_values(frame, indices, "battery_temperature", 2.5 * scale * progress)
    elif anomaly_type == "solar_array_underperformance":
        channels = ("solar_array_current", "solar_array_voltage")
        subsystem = "EPS"
        _set_values(frame, indices, "solar_array_current", -1.8 * scale * progress)
        _set_values(frame, indices, "solar_array_voltage", -3.5 * scale * progress)
    elif anomaly_type == "heater_stuck_on":
        channels = ("heater_state", "radiator_temperature", "panel_temperature")
        subsystem = "THERMAL"
        _set_values(frame, indices, "heater_state", 1.0, operation="set")
        _set_values(frame, indices, "radiator_temperature", 7.0 * scale * progress)
        _set_values(frame, indices, "panel_temperature", 3.0 * scale * progress)
    elif anomaly_type == "heater_stuck_off":
        channels = ("heater_state", "radiator_temperature", "battery_temperature")
        subsystem = "THERMAL"
        _set_values(frame, indices, "heater_state", 0.0, operation="set")
        _set_values(frame, indices, "radiator_temperature", -7.0 * scale * progress)
        _set_values(frame, indices, "battery_temperature", -2.5 * scale * progress)
    elif anomaly_type == "payload_overcurrent":
        channels = ("payload_current", "bus_current", "bus_voltage")
        subsystem = "PAYLOAD"
        _set_values(frame, indices, "payload_current", 2.2 * scale * progress)
        _set_values(frame, indices, "bus_current", 2.0 * scale * progress)
        _set_values(frame, indices, "bus_voltage", -0.8 * scale * progress)
    elif anomaly_type == "communication_link_margin_drop":
        channels = ("link_margin", "receiver_rssi", "antenna_temperature")
        subsystem = "COMM"
        _set_values(frame, indices, "link_margin", -8.0 * scale * progress)
        _set_values(frame, indices, "receiver_rssi", -6.0 * scale * progress)
        _set_values(frame, indices, "antenna_temperature", 2.0 * scale * progress)
    elif anomaly_type == "sensor_bias_drift":
        channels = ("gyro_rate_y", "attitude_error")
        subsystem = "AOCS"
        _set_values(frame, indices, "gyro_rate_y", 0.09 * scale * progress)
        _set_values(frame, indices, "attitude_error", 0.08 * scale * progress)
    elif anomaly_type == "battery_voltage_sag":
        channels = ("battery_voltage", "battery_current", "bus_voltage")
        subsystem = "EPS"
        _set_values(frame, indices, "battery_voltage", -1.5 * scale * progress)
        _set_values(frame, indices, "battery_current", 1.3 * scale * progress)
        _set_values(frame, indices, "bus_voltage", -0.7 * scale * progress)
    elif anomaly_type == "thermal_runaway":
        channels = (
            "payload_temperature",
            "panel_temperature",
            "radiator_temperature",
        )
        subsystem = "THERMAL"
        _set_values(frame, indices, "payload_temperature", 11.0 * scale * progress**2)
        _set_values(frame, indices, "panel_temperature", 6.0 * scale * progress**2)
        _set_values(frame, indices, "radiator_temperature", 4.0 * scale * progress**2)
    else:
        raise ValueError(f"unsupported anomaly type: {anomaly_type}")

    event_id = f"SYN-{event_number:04d}"
    precursor = np.arange(early_start, onset)
    anomaly = np.arange(onset, critical)
    critical_region = np.arange(critical, end)
    for region, label in (
        (precursor, "precursor"),
        (anomaly, "anomaly"),
        (critical_region, "critical"),
    ):
        frame.iloc[region, frame.columns.get_loc("is_anomaly")] = True
        frame.iloc[region, frame.columns.get_loc("anomaly_event_id")] = event_id
        frame.iloc[region, frame.columns.get_loc("anomaly_type")] = anomaly_type
        frame.iloc[region, frame.columns.get_loc("label_taxonomy")] = label

    return InjectionRecord(
        event_id=event_id,
        partition=partition,
        anomaly_type=anomaly_type,
        severity=severity,
        start=frame.index[onset],
        end=frame.index[end - 1],
        affected_channels=channels,
        expected_subsystem=subsystem,
        early_warning_region_start=frame.index[early_start],
        failure_region_start=frame.index[critical],
        event_class="anomaly",
        notes="Controlled synthetic anomaly with precursor and critical regions.",
        parameters=parameters,
    )


def _inject_benign(
    frame: pd.DataFrame,
    *,
    benign_type: str,
    partition: str,
    start: int,
    partition_end: int,
    event_number: int,
) -> InjectionRecord:
    duration = 24
    end = min(start + duration, partition_end)
    indices = np.arange(start, end)
    progress = np.sin(np.linspace(0.0, np.pi, len(indices)))
    if benign_type == "mode_transition_transient":
        channels = ("payload_current", "bus_current")
        subsystem = "PAYLOAD"
        _set_values(frame, indices, "payload_current", 0.35 * progress)
        _set_values(frame, indices, "bus_current", 0.25 * progress)
    elif benign_type == "eclipse_thermal_transient":
        channels = ("panel_temperature", "radiator_temperature")
        subsystem = "THERMAL"
        _set_values(frame, indices, "panel_temperature", -1.2 * progress)
        _set_values(frame, indices, "radiator_temperature", -1.8 * progress)
    elif benign_type == "safe_mode_transition":
        channels = ("transmitter_power", "payload_current")
        subsystem = "COMM"
        _set_values(frame, indices, "transmitter_power", -0.8 * progress)
        _set_values(frame, indices, "payload_current", -0.15 * progress)
        frame.iloc[indices, frame.columns.get_loc("safe_mode_flag")] = True
    else:
        raise ValueError(f"unsupported benign event: {benign_type}")

    event_id = f"SYN-{event_number:04d}"
    frame.iloc[indices, frame.columns.get_loc("anomaly_event_id")] = event_id
    frame.iloc[indices, frame.columns.get_loc("anomaly_type")] = benign_type
    frame.iloc[indices, frame.columns.get_loc("label_taxonomy")] = "benign_transient"
    return InjectionRecord(
        event_id=event_id,
        partition=partition,
        anomaly_type=benign_type,
        severity="nominal",
        start=frame.index[start],
        end=frame.index[end - 1],
        affected_channels=channels,
        expected_subsystem=subsystem,
        early_warning_region_start=frame.index[start],
        failure_region_start=frame.index[end - 1],
        event_class="benign_transient",
        notes="Benign operational transient; alarms should be suppressed.",
        parameters={},
    )


def _scheduled_events(
    frame: pd.DataFrame,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[InjectionRecord, ...]:
    schedule = config.anomaly_schedule or DEFAULT_ANOMALY_SCHEDULE
    bounds = _partition_bounds(config)
    events: list[InjectionRecord] = []
    event_number = 1
    for partition in ("calibration", "validation", "test"):
        partition_config = schedule.get(
            partition,
            schedule.get(f"{partition}_events", {}),
        )
        if not bool(partition_config.get("enabled", True)):
            continue
        count_per_type = int(partition_config.get("count_per_type", 1))
        severity_levels = tuple(
            str(value) for value in partition_config.get("severity_levels", ["medium"])
        )
        if not severity_levels or any(level not in SEVERITY_SCALE for level in severity_levels):
            raise ValueError(f"invalid severity levels for {partition}")
        event_specs = [
            (anomaly_type, instance)
            for anomaly_type in ANOMALY_TYPES
            for instance in range(count_per_type)
        ]
        benign_count = max(0, int(config.benign_events_per_partition))
        total_slots = len(event_specs) + benign_count
        partition_start, partition_end = bounds[partition]
        spacing = (partition_end - partition_start) / (total_slots + 1)
        if spacing < 90:
            raise ValueError(
                f"{partition} is too short for {total_slots} scheduled events"
            )
        for slot, (anomaly_type, instance) in enumerate(event_specs, start=1):
            severity = severity_levels[(slot + instance - 1) % len(severity_levels)]
            early_start = int(partition_start + slot * spacing - 35)
            events.append(
                _inject_anomaly(
                    frame,
                    anomaly_type=anomaly_type,
                    partition=partition,
                    severity=severity,
                    early_start=early_start,
                    partition_end=partition_end,
                    rng=rng,
                    event_number=event_number,
                )
            )
            event_number += 1
        for offset in range(benign_count):
            slot = len(event_specs) + offset + 1
            benign_type = BENIGN_TYPES[offset % len(BENIGN_TYPES)]
            start = int(partition_start + slot * spacing - 12)
            events.append(
                _inject_benign(
                    frame,
                    benign_type=benign_type,
                    partition=partition,
                    start=start,
                    partition_end=partition_end,
                    event_number=event_number,
                )
            )
            event_number += 1
    return tuple(events)


def _add_missing_values(
    frame: pd.DataFrame,
    channels: tuple[str, ...],
    fraction: float,
    rng: np.random.Generator,
) -> None:
    if fraction <= 0.0:
        return
    nominal_rows = np.flatnonzero(~frame["is_anomaly"].to_numpy(dtype=bool))
    count = int(len(nominal_rows) * len(channels) * fraction)
    if count == 0:
        return
    rows = rng.choice(nominal_rows, size=count, replace=True)
    columns = rng.integers(0, len(channels), size=count)
    for row, column in zip(rows, columns, strict=True):
        frame.iat[int(row), frame.columns.get_loc(channels[int(column)])] = np.nan


def generate_synthetic_telemetry(config: SyntheticConfig | None = None) -> SyntheticDataset:
    """Generate telemetry with held-out partition-aware event instances."""

    config = config or SyntheticConfig()
    if config.periods < 12000:
        raise ValueError("periods must be at least 12000 for the anomaly schedule")
    if config.benign_events_per_partition < 0:
        raise ValueError("benign_events_per_partition cannot be negative")
    rng = np.random.default_rng(config.seed)
    frame = _nominal_frame(config, rng)
    events = _scheduled_events(frame, config, rng)
    _add_missing_values(frame, TELEMETRY_CHANNELS, config.missing_fraction, rng)
    return SyntheticDataset(frame=frame, events=events)
