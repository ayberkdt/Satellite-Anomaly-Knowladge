from sak.visualization.theme import SUBSYSTEM_COLORS, is_hex_color, subsystem_color


def test_known_subsystems_have_stable_hex_colors() -> None:
    for subsystem in ("EPS", "THERMAL", "AOCS", "COMM", "PAYLOAD", "UNKNOWN"):
        assert subsystem in SUBSYSTEM_COLORS
        assert is_hex_color(SUBSYSTEM_COLORS[subsystem])


def test_unknown_and_adcs_fallback_colors_are_stable() -> None:
    assert subsystem_color("ADCS") == SUBSYSTEM_COLORS["AOCS"]
    assert subsystem_color("not-a-real-subsystem") == SUBSYSTEM_COLORS["UNKNOWN"]
