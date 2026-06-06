"""Render model-independent SAK reports as Markdown."""

from __future__ import annotations

from sak.contracts import AlarmEvent


def render_early_warning_report(event: AlarmEvent) -> str:
    """Render an alarm event into the standard engineering report template."""

    explanation = event.explanation
    if explanation is None:
        channel_lines = "- Açıklama henüz üretilmedi."
        critical_window = "Belirlenmedi"
        subsystem = "Belirlenmedi"
        interpretation = "Model açıklaması bekleniyor."
        confidence = "Belirlenmedi"
    else:
        channel_lines = "\n".join(
            f"- {item.channel}: {item.contribution:.3f}"
            + (f" ({item.subsystem})" if item.subsystem else "")
            + (f" [{item.direction}]" if item.direction else "")
            for item in explanation.contributions
        )
        critical_window = (
            f"{explanation.critical_start.isoformat()} — "
            f"{explanation.critical_end.isoformat()}"
        )
        subsystem = ", ".join(explanation.possible_subsystems) or "Belirlenmedi"
        interpretation = " ".join(explanation.notes) or "Uzman yorumu gerekli."
        confidence = (
            f"{explanation.confidence:.1%}"
            if explanation.confidence is not None
            else "Belirlenmedi"
        )
    context = ", ".join(f"{key}={value}" for key, value in event.context.items())
    context = context or "Belirlenmedi"

    return f"""# SAK Early Warning Report

- **Event ID:** {event.event_id}
- **Alarm time:** {event.peak_time.isoformat()}
- **Event interval:** {event.start_time.isoformat()} — {event.end_time.isoformat()}
- **Anomaly score:** {event.peak_score:.4f}
- **Threshold:** {event.threshold:.4f}
- **Risk level:** {event.risk_level}
- **Operational context:** {context}
- **Critical time window:** {critical_window}
- **Possible subsystem:** {subsystem}
- **Attribution concentration:** {confidence}
- **Model confidence / uncertainty:** Henüz kalibre edilmedi

## Top Contributing Telemetry Channels

{channel_lines}

## Engineering Interpretation

{interpretation}

## Suggested Next Inspection

1. İlgili alt sistemin limit ve durum telemetrilerini doğrula.
2. Aynı operasyon modu ve yörünge fazındaki nominal örneklerle karşılaştır.
3. Komut, olay ve bakım kayıtlarını kritik pencereyle zaman hizalı incele.
"""
