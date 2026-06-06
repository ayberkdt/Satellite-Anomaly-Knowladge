# Dashboards

Dashboard, model geliştirme katmanından ayrıdır. İlk sürümde aşağıdaki
görünümler hedeflenir:

- Ham telemetri ve operasyon bağlamı
- Ham ve filtrelenmiş anomaly score
- Alarm/event zaman çizelgesi
- Kanal-zaman katkı heatmap'i
- Alt sistem ve mühendislik yorum kartı

## Mevcut dashboard

İlk çalışan UI çıktısı:

```text
dashboards/sak_synthetic_dashboard.html
```

Bu dosya `experiments/run_synthetic_models.py` çalıştırıldığında otomatik
yenilenir. Statik HTML olduğu için ek web framework gerektirmez; dosyayı
tarayıcıda açmak yeterlidir.

Dashboard şu bölümleri içerir:

- Overview kartları
- Model karşılaştırma grafikleri
- Threshold sweep tablosu ve grafiği
- Detection delay grafiği
- Event diagnostics: anomaly type, delay, channel hit, subsystem hit
- False positive event tablosu
- Top explanation channel ve subsystem contribution grafikleri
- PCA score timeline
- Dense Autoencoder channel-error heatmap
- Event/XAI kartları
- CSV ve Markdown sonuç artefact bağlantıları

Üretilen tablo ve grafikler:

```text
artifacts/synthetic_models/dashboard/
├── model_comparison.csv
├── threshold_sweep.csv
├── detection_delays.csv
├── event_diagnostics.csv
├── false_positives.csv
├── channel_summary.csv
├── subsystem_summary.csv
├── results_summary.md
├── model_comparison.png
├── false_alarm_delay.png
├── threshold_sweep.png
├── detection_delays.png
├── event_diagnostics.png
├── channel_summary.png
└── subsystem_summary.png
```
