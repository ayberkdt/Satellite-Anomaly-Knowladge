"""Physics-inspired synthetic satellite telemetry for pipeline validation."""

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
    "solar_array_voltage",
    "solar_array_current",
    "bus_voltage",
    "panel_temperature",
    "payload_temperature",
    "angular_rate_x",
    "reaction_wheel_speed",
    "transmitter_power",
    "receiver_rssi",
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for nominal telemetry and default anomaly injections."""

    periods: int = 14 * 24 * 60
    frequency: str = "1min"
    start: str = "2026-01-01T00:00:00Z"
    orbit_period_steps: int = 96
    test_fraction: float = 0.20
    seed: int = 42
    missing_fraction: float = 0.001


@dataclass(frozen=True)
class InjectionRecord:
    """Ground-truth metadata for one synthetic anomaly event."""

    event_id: str
    anomaly_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    affected_channels: tuple[str, ...]
    expected_subsystem: str
    parameters: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start"] = self.start.isoformat()
        payload["end"] = self.end.isoformat()
        payload["affected_channels"] = list(self.affected_channels)
        return payload


@dataclass(frozen=True)
class SyntheticDataset:
    """Generated telemetry frame and its exact injection manifest."""

    frame: pd.DataFrame
    events: tuple[InjectionRecord, ...]
    channel_names: tuple[str, ...] = TELEMETRY_CHANNELS


def _ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _nominal_frame(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    start = pd.Timestamp(config.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    timestamps = pd.date_range(
        start=start,
        periods=config.periods,
        freq=config.frequency,
    )
    step = np.arange(config.periods)
    orbit_phase = (step % config.orbit_period_steps) / config.orbit_period_steps
    sunlight = orbit_phase < 0.62
    eclipse = ~sunlight

    payload_cycle = (step // 240) % 3
    operational_mode = np.where(payload_cycle == 1, "payload", "nominal").astype(object)
    safe_mask = (step % (4 * 24 * 60)) < 25
    operational_mode[safe_mask] = "safe"
    payload_active = operational_mode == "payload"

    orbit_wave = np.sin(2.0 * np.pi * orbit_phase)
    solar_array_current = (
        sunlight * (4.7 + 0.35 * orbit_wave) + rng.normal(0.0, 0.07, config.periods)
    )
    solar_array_current = np.clip(solar_array_current, 0.0, None)
    solar_array_voltage = (
        sunlight * (31.0 + 0.45 * orbit_wave)
        + eclipse * 0.6
        + rng.normal(0.0, 0.08, config.periods)
    )

    payload_load = 1.6 * payload_active + 0.15 * (operational_mode == "safe")
    base_load = 1.15 + payload_load + rng.normal(0.0, 0.05, config.periods)
    battery_current = base_load - 0.72 * solar_array_current

    state_of_charge = np.empty(config.periods, dtype=float)
    state_of_charge[0] = 78.0
    for index in range(1, config.periods):
        delta = -battery_current[index] * 0.010
        state_of_charge[index] = np.clip(state_of_charge[index - 1] + delta, 48.0, 96.0)

    thermal_driver = 0.8 * sunlight + 0.35 * payload_active
    thermal_state = _ewma(thermal_driver.astype(float), alpha=0.035)
    panel_temperature = (
        4.0
        + 30.0 * thermal_state
        + 2.2 * orbit_wave
        + rng.normal(0.0, 0.28, config.periods)
    )
    payload_temperature = (
        15.0
        + 8.5 * _ewma(payload_active.astype(float), alpha=0.025)
        + 0.16 * panel_temperature
        + rng.normal(0.0, 0.20, config.periods)
    )
    battery_temperature = (
        18.0
        + 0.18 * panel_temperature
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
        - 0.055 * np.maximum(base_load, 0.0)
        + 0.035 * sunlight
        + rng.normal(0.0, 0.025, config.periods)
    )

    angular_rate_x = (
        0.015 * np.sin(4.0 * np.pi * orbit_phase)
        + rng.normal(0.0, 0.004, config.periods)
    )
    reaction_wheel_speed = (
        1850.0
        + 95.0 * np.sin(2.0 * np.pi * orbit_phase + 0.5)
        + 140.0 * payload_active
        + rng.normal(0.0, 8.0, config.periods)
    )
    transmitter_power = (
        6.0
        + 2.5 * payload_active
        - 1.5 * (operational_mode == "safe")
        + rng.normal(0.0, 0.08, config.periods)
    )
    receiver_rssi = (
        -72.0
        + 2.8 * np.cos(2.0 * np.pi * orbit_phase)
        + 0.25 * transmitter_power
        + rng.normal(0.0, 0.35, config.periods)
    )

    return pd.DataFrame(
        {
            "battery_voltage": battery_voltage,
            "battery_current": battery_current,
            "battery_temperature": battery_temperature,
            "battery_state_of_charge": state_of_charge,
            "solar_array_voltage": solar_array_voltage,
            "solar_array_current": solar_array_current,
            "bus_voltage": bus_voltage,
            "panel_temperature": panel_temperature,
            "payload_temperature": payload_temperature,
            "angular_rate_x": angular_rate_x,
            "reaction_wheel_speed": reaction_wheel_speed,
            "transmitter_power": transmitter_power,
            "receiver_rssi": receiver_rssi,
            "operational_mode": operational_mode,
            "eclipse": eclipse,
            "sunlight": sunlight,
            "orbit_phase": orbit_phase,
            "is_anomaly": False,
            "anomaly_event_id": "",
            "anomaly_type": "",
        },
        index=timestamps,
    ).rename_axis("timestamp")


def _default_schedule(test_start: int) -> tuple[tuple[str, int, int], ...]:
    return (
        ("spike", test_start + 120, 10),
        ("drift", test_start + 350, 180),
        ("step_change", test_start + 700, 120),
        ("slow_degradation", test_start + 1000, 150),
        ("stuck_sensor", test_start + 1350, 120),
        ("noise_increase", test_start + 1650, 120),
        ("correlation_break", test_start + 1950, 160),
        ("thermal_runaway", test_start + 2300, 120),
        ("voltage_drop_current_rise", test_start + 2650, 100),
        ("orbit_dependent_thermal", test_start + 3050, 300),
    )


def _inject(
    frame: pd.DataFrame,
    anomaly_type: str,
    start: int,
    duration: int,
    rng: np.random.Generator,
    event_number: int,
) -> InjectionRecord:
    end = min(start + duration, len(frame))
    index = np.arange(start, end)
    progress = np.linspace(0.0, 1.0, len(index), endpoint=True)
    parameters: dict[str, float]

    if anomaly_type == "spike":
        channels = ("bus_voltage",)
        subsystem = "EPS"
        frame.iloc[index, frame.columns.get_loc("bus_voltage")] += 2.4
        parameters = {"delta": 2.4}
    elif anomaly_type == "drift":
        channels = ("battery_temperature",)
        subsystem = "EPS"
        frame.iloc[index, frame.columns.get_loc("battery_temperature")] += 6.0 * progress
        parameters = {"final_delta": 6.0}
    elif anomaly_type == "step_change":
        channels = ("receiver_rssi",)
        subsystem = "COMM"
        frame.iloc[index, frame.columns.get_loc("receiver_rssi")] -= 7.0
        parameters = {"delta": -7.0}
    elif anomaly_type == "slow_degradation":
        channels = ("battery_state_of_charge", "battery_voltage")
        subsystem = "EPS"
        degradation = 9.0 * progress**2
        frame.iloc[index, frame.columns.get_loc("battery_state_of_charge")] -= degradation
        frame.iloc[index, frame.columns.get_loc("battery_voltage")] -= 0.08 * degradation
        parameters = {"soc_final_delta": -9.0, "voltage_final_delta": -0.72}
    elif anomaly_type == "stuck_sensor":
        channels = ("solar_array_current",)
        subsystem = "EPS"
        stuck_value = float(frame.iloc[start]["solar_array_current"])
        frame.iloc[index, frame.columns.get_loc("solar_array_current")] = stuck_value
        parameters = {"stuck_value": stuck_value}
    elif anomaly_type == "noise_increase":
        channels = ("angular_rate_x",)
        subsystem = "AOCS"
        noise = rng.normal(0.0, 0.055, len(index))
        frame.iloc[index, frame.columns.get_loc("angular_rate_x")] += noise
        parameters = {"noise_std": 0.055}
    elif anomaly_type == "correlation_break":
        channels = ("bus_voltage", "battery_current")
        subsystem = "EPS"
        replacement = rng.normal(
            float(frame["bus_voltage"].median()),
            0.22,
            len(index),
        )
        frame.iloc[index, frame.columns.get_loc("bus_voltage")] = replacement
        parameters = {"replacement_std": 0.22}
    elif anomaly_type == "thermal_runaway":
        channels = ("payload_temperature", "panel_temperature")
        subsystem = "THERMAL"
        runaway = 12.0 * progress**2
        frame.iloc[index, frame.columns.get_loc("payload_temperature")] += runaway
        frame.iloc[index, frame.columns.get_loc("panel_temperature")] += 0.55 * runaway
        parameters = {"payload_final_delta": 12.0, "panel_final_delta": 6.6}
    elif anomaly_type == "voltage_drop_current_rise":
        channels = ("battery_voltage", "battery_current", "battery_temperature")
        subsystem = "EPS"
        frame.iloc[index, frame.columns.get_loc("battery_voltage")] -= 1.4
        frame.iloc[index, frame.columns.get_loc("battery_current")] += 1.8
        frame.iloc[index, frame.columns.get_loc("battery_temperature")] += 2.5 * progress
        parameters = {"voltage_delta": -1.4, "current_delta": 1.8}
    elif anomaly_type == "orbit_dependent_thermal":
        channels = ("panel_temperature", "payload_temperature")
        subsystem = "THERMAL"
        sunlight = frame.iloc[index]["sunlight"].to_numpy(dtype=bool)
        frame.iloc[index, frame.columns.get_loc("panel_temperature")] += 7.0 * sunlight
        frame.iloc[index, frame.columns.get_loc("payload_temperature")] += 3.0 * sunlight
        parameters = {"panel_sunlight_delta": 7.0, "payload_sunlight_delta": 3.0}
    else:
        raise ValueError(f"Unsupported anomaly type: {anomaly_type}")

    event_id = f"SYN-{event_number:04d}"
    frame.iloc[index, frame.columns.get_loc("is_anomaly")] = True
    frame.iloc[index, frame.columns.get_loc("anomaly_event_id")] = event_id
    frame.iloc[index, frame.columns.get_loc("anomaly_type")] = anomaly_type

    return InjectionRecord(
        event_id=event_id,
        anomaly_type=anomaly_type,
        start=frame.index[start],
        end=frame.index[end - 1],
        affected_channels=channels,
        expected_subsystem=subsystem,
        parameters=parameters,
    )


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
    """Generate nominal telemetry and inject deterministic test-only anomalies."""

    config = config or SyntheticConfig()
    if config.periods < 18000:
        raise ValueError("periods must be at least 18000 for the default anomaly schedule")
    if not 0.0 < config.test_fraction < 0.5:
        raise ValueError("test_fraction must be in (0, 0.5)")

    rng = np.random.default_rng(config.seed)
    frame = _nominal_frame(config, rng)
    test_start = int(config.periods * (1.0 - config.test_fraction))
    events = tuple(
        _inject(frame, anomaly_type, start, duration, rng, event_number)
        for event_number, (anomaly_type, start, duration) in enumerate(
            _default_schedule(test_start),
            start=1,
        )
    )
    _add_missing_values(frame, TELEMETRY_CHANNELS, config.missing_fraction, rng)
    return SyntheticDataset(frame=frame, events=events)
