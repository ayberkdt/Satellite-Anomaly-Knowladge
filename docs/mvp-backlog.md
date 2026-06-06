# İlk MVP, Dört Haftalık Plan ve Riskler

## 1. MVP tanımı

İlk uygulanabilir prototip şu uçtan uca akışı sağlamalıdır:

```text
CSV/Parquet
  -> validation/preprocessing
  -> chronological split/windowing
  -> PCA or Autoencoder
  -> anomaly score
  -> calibrated threshold
  -> EWMA + persistence
  -> event
  -> top channels + heatmap
  -> SAK Early Warning Report
```

MVP, GNN, Transformer veya tam teşekküllü web dashboard içermez.

## 2. Net yapılacaklar listesi

### Veri ve şema

- [ ] Kanonik telemetry schema ve kanal kataloğu tanımla.
- [ ] CSV/Parquet adaptörünü schema validation ile tamamla.
- [ ] Dataset manifest ve checksum üret.
- [ ] Timestamp, duplicate, missingness ve unit kontrollerini ekle.
- [ ] Kronolojik split'i interval düzeyinde uygula.
- [ ] Train-only scaler fit ve serialization ekle.
- [ ] Causal resampling/interpolation kurallarını uygula.
- [ ] Window builder ve validity/imputation maskelerini ekle.

### Sentetik veri

- [x] Orbit, eclipse, mode, EPS ve thermal ilişkilerini simüle et.
- [x] En az spike, drift, step, stuck sensor ve combined EPS injection ekle.
- [x] Injection manifest üret.
- [x] Seed reproducibility testi yaz.
- [x] Nominal/anomaly görselleri üret.

### PCA baseline

- [x] `AnomalyModel` arayüzüne PCA implementasyonu ekle.
- [x] Explained variance'i yalnız train ile seç.
- [x] Aggregate ve channel-level reconstruction error üret.
- [x] Model/scaler/channel-order checkpoint'i kaydet.
- [ ] Validation quantile ve MAD threshold seçeneklerini ekle.

### Autoencoder

- [x] Config-driven Dense Autoencoder kur.
- [x] Training/validation loop ve early stopping ekle.
- [x] Seed, device, checkpoint ve loss logging ekle.
- [x] Aggregate ve channel-time reconstruction error üret.
- [x] PCA ile aynı evaluation API'sini kullan.

### Alarm mantığı

- [x] EWMA ve m-of-n persistence çekirdeği.
- [x] Alarm noktalarını event interval'larına birleştir.
- [ ] Cooldown ve merge tolerance testlerini genişlet.
- [ ] Mode-conditioned threshold opsiyonu ekle.
- [ ] Risk level kurallarını config'e taşı.

### XAI ve raporlama

- [x] Reconstruction error contribution normalizasyonu.
- [x] Top-k kanal sıralaması.
- [x] Kanal-subsystem mapping loader.
- [x] Temporal critical window seçimi.
- [x] Kanal-zaman heatmap üretimi.
- [x] Modelden bağımsız Markdown rapor şablonu.
- [ ] JSON report export ve schema testi.
- [x] Sentetik ground truth için Hit@k/temporal IoU.

### Değerlendirme ve kalite

- [x] Point-wise metrikleri ekle.
- [x] One-to-one event matching ve event metrikleri ekle.
- [x] Detection delay ve false alarms/day-orbit ekle.
- [ ] Config snapshot ve run manifest oluştur.
- [ ] Unit, integration ve leakage regression testleri yaz.
- [ ] Ruff, mypy ve pytest CI job'u ekle.

## 3. MVP kabul kriterleri

- Tek config ile sentetik veriden rapora kadar pipeline çalışır.
- CSV ve Parquet aynı kanonik sonucu verir.
- Test partition hiçbir fit/calibration adımında kullanılmaz.
- PCA ve Dense AE aynı score/evaluation sözleşmesini uygular.
- Enjekte edilen anomaly event'i raporda zaman ve top-k kanal ile görünür.
- Persistence filtresi tekil noise spike'larını alarm yapmaz.
- Point/event metrikleri, delay ve false alarm birlikte raporlanır.
- Sonuç aynı seed ve veri sürümünde tekrar üretilebilir.

İlk sayısal hedefler veri görülmeden proje başarısı olarak dondurulmamalıdır.
Sentetik başlangıç hedefi olarak event recall ≥ 0.80, false alarm ≤ 1/gün ve
channel Hit@3 ≥ 0.80 izlenebilir; gerçek veri sonrası bu bütçeler yeniden
kalibre edilir.

## 4. İlk dört haftalık çalışma planı

### Hafta 1 — Veri sözleşmesi ve sentetik temel

**Hedef**

Güvenilir veri yolunu kurmak.

**Teslimatlar**

- Kanonik schema ve kanal kataloğu.
- CSV/Parquet loader validation.
- Kronolojik split ve scaler.
- İlk orbit/eclipse/mode-aware sentetik dataset.
- Veri kalite ve leakage testleri.

**Hafta sonu demosu**

Bir sentetik dosyanın okunması, split edilmesi ve telemetri/context grafiğinin
üretilmesi.

### Hafta 2 — PCA, skor ve event değerlendirmesi

**Hedef**

İlk açıklanabilir baseline ve ölçüm protokolü.

**Teslimatlar**

- PCA reconstruction modeli.
- Threshold calibration.
- EWMA, persistence ve event builder.
- Point/event metrikleri, delay, false alarms/day.
- Top-k residual channel açıklaması.

**Hafta sonu demosu**

Bir injection için skor zaman çizgisi, yakalanan event ve top-3 kanal.

### Hafta 3 — Dense Autoencoder ve heatmap

**Hedef**

Doğrusal olmayan modelin baseline üstü katkısını ölçmek.

**Teslimatlar**

- PyTorch Dense AE.
- Training loop, checkpoint ve early stopping.
- Channel-time reconstruction heatmap.
- PCA/AE karşılaştırma tablosu.
- En az üç seed deneyi.

**Hafta sonu demosu**

Aynı test setinde PCA ve AE event/false alarm/delay karşılaştırması.

### Hafta 4 — Raporlama, sağlamlaştırma ve araştırma paketi

**Hedef**

Uçtan uca MVP ve sunulabilir araştırma çıktısı.

**Teslimatlar**

- Subsystem mapping ve engineering interpretation kuralları.
- Markdown/JSON Early Warning Report.
- Otomatik deney manifesti ve grafik paketi.
- Integration test ve leakage regression testleri.
- İlk sonuç bölümü, limitler ve sonraki deney planı.

**Hafta sonu demosu**

CSV/Parquet girdiden otomatik rapora tek komutluk akış.

## 5. Risk kaydı ve azaltma planı

| Risk | Etki | Azaltma |
|---|---|---|
| Az/yanlış anomaly etiketi | Metrikler yanıltıcı olur | Event toleransı, uzman incelemesi, sentetik kontrollü test |
| Yüksek false alarm | Operasyonel güven kaybı | Mode threshold, persistence, cooldown, false alarm bütçesi |
| Veri sızıntısı | Gerçek dışı yüksek performans | Kronolojik split, train-only fit, leakage testleri |
| Mode değişimi anomaly görünür | Yanlış alarm | Context feature, mode-specific model/threshold |
| Missing telemetry | Sahte residual | Mask, sınırlı interpolation, validity filtering |
| AE anomaly'yi reconstruct eder | Düşük recall | Bottleneck, regularization, nominal-only/robust training |
| XAI korelasyonu neden sanır | Hatalı mühendislik yorumu | Belirsizlik dili, occlusion, fiziksel bilgiyle kontrol |
| Çok kanallı subsystem baskın çıkar | Yanlış subsystem sırası | Kanal sayısı ve nominal hata ile normalize et |
| Gerçek veri sentetikten farklı | Genelleme düşer | Domain adapter, transfer calibration, gerçek EDA |
| GNN erken karmaşıklık yaratır | Süre ve açıklama kaybı | v4 geçiş kriterleri, baseline ablation zorunluluğu |
| Tek metrik optimizasyonu | Kullanışsız model seçimi | Event, delay, false alarm, XAI ve maliyet Pareto analizi |
| Yetersiz reproducibility | Sonuçlar savunulamaz | Seed, config, checksum, checkpoint, run manifest |

## 6. Örnek SAK Early Warning Report

```markdown
# SAK Early Warning Report

- Alarm time: 2026-06-06T10:42:00Z
- Event interval: 2026-06-06T10:37:00Z — 2026-06-06T10:49:00Z
- Anomaly score: 4.82
- Threshold: 2.35
- Risk level: HIGH
- Operational context: eclipse / nominal attitude mode
- Critical time window: 2026-06-06T10:39:00Z — 2026-06-06T10:44:00Z
- Possible subsystem: EPS
- Confidence / uncertainty: Medium (0.76)

## Top Contributing Telemetry Channels

1. battery_voltage: 0.41, decreasing
2. battery_current: 0.29, increasing
3. battery_temperature: 0.13, increasing

## Engineering Interpretation

Eclipse sırasında beklenen batarya gerilim-akım ilişkisi nominal örneklerden
ayrışmıştır. Gerilim düşüşüne akım ve sıcaklık artışı eşlik etmektedir. Bulgular
EPS kaynaklı olası bir güç/batarya problemini işaret eder; kesin arıza teşhisi
değildir.

## Suggested Next Inspection

1. State-of-charge ve batarya hücre gerilimlerini kontrol et.
2. Aynı orbit fazındaki son nominal eclipse geçişleriyle karşılaştır.
3. Event öncesi payload yükü ve telecommand kayıtlarını incele.
```

## 7. Definition of done

Bir görev yalnız kod merge edildiğinde değil, şu koşullarda tamamlanır:

- Testi vardır.
- Config ile kontrol edilir.
- Kullanılan split ve veri sürümü loglanır.
- Metrik veya artefact üretir.
- Leakage ve causal kullanım açısından gözden geçirilmiştir.
- Araştırma raporunda hangi soruyu yanıtladığı bellidir.
