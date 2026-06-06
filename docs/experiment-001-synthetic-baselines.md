# Deney 001 — Sentetik PCA ve Dense Autoencoder

## Amaç

SAK veri, model, alarm, event değerlendirme ve açıklama katmanlarının ilk kez
uçtan uca çalıştırılması.

## Deney düzeni

- 14 gün, 1 dakika çözünürlük, 20.160 satır.
- 13 sayısal telemetri kanalı.
- Orbit phase, eclipse/sunlight ve operational mode bağlamı.
- Kronolojik bölme: `%60` train, `%20` validation, `%20` test.
- Train ve validation tamamen nominal.
- Test bölümünde 10 kontrollü anomaly event'i.
- Robust scaling yalnız train üzerinde fit edildi.
- Threshold yalnız nominal validation EWMA skorundan seçildi.
- Alarm: EWMA + `3-of-5` persistence.

Injection türleri:

1. Spike
2. Drift
3. Step change
4. Slow degradation
5. Stuck sensor
6. Noise increase
7. Correlation break
8. Thermal runaway
9. Voltage drop + current rise
10. Orbit-dependent thermal anomaly

## İlk sonuç

| Metrik | PCA | Dense AE |
|---|---:|---:|
| Event precision | 0.909 | 0.909 |
| Event recall | 1.000 | 1.000 |
| Event F1 | 0.952 | 0.952 |
| False alarms/day | 0.357 | 0.357 |
| Median delay | 13.5 dk | 13.5 dk |
| Point F1 | 0.655 | 0.714 |
| Channel Hit@3 | 0.900 | 1.000 |
| Subsystem Hit@2 | 0.900 | 1.000 |
| Critical-window hit rate | 0.900 | 1.000 |

Bu değerler sentetik veri üzerindedir ve gerçek görev performansı iddiası
değildir.

## Yorum

- Her iki model 10 event'in tamamını yakaladı.
- Dense AE anomaly interval'larının daha büyük bölümünü işaretledi ve kanal
  sıralamasında PCA'dan iyi sonuç verdi.
- PCA yalnız beş principal component ile varyansın yaklaşık `%97.7` bölümünü
  korudu; buna rağmen event sonucu AE ile aynı kaldı. Bu, PCA'nın güçlü ve
  vazgeçilmemesi gereken bir baseline olduğunu gösteriyor.
- Her model bir ek event üretti. Bu false alarm payload operasyon modu
  geçişinde oluştu. Sonraki deney mode-aware threshold veya context-conditioned
  model olmalıdır.
- Önceden tanımlı threshold taramasında PCA `%99.9` validation quantile ile bu
  sentetik testte `event F1 = 1.0` ve `false alarm/day = 0` verdi. Bu değer test
  sonucu görüldükten sonra varsayılan yapılmayacak; yeni seed veya ayrı
  senaryoda doğrulanacaktır.
- Slow degradation, drift ve thermal runaway olaylarında gecikme yüksektir.
  Persistence ayarını gevşetmek tek başına yeterli olmayabilir; slope/trend
  skoru ve temporal model denenmelidir.
- Critical-window IoU uzun event'lere karşı düşük görünür; kritik pencere tüm
  olayı kaplamak için değil, en etkili alt aralığı bulmak için tasarlanmıştır.
  Bu nedenle ayrıca critical-window hit rate raporlanır.

## Üretilen artefact'lar

- `data/synthetic/telemetry.csv`
- `data/synthetic/injection_manifest.json`
- `artifacts/synthetic_models/comparison.json`
- Her model için checkpoint, skor CSV'si, event JSON'u, timeline ve heatmap.
- Her tahmin event'i için SAK Early Warning Report.

## Sonraki deney

1. Validation üzerinden threshold quantile hassasiyet taraması.
2. Operational-mode bazlı threshold.
3. Drift/slope ve rate-of-change baseline.
4. TCN veya LSTM sequence autoencoder.
5. Birden fazla seed ile ortalama ve standart sapma.
