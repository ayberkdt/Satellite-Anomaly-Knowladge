# Modelleme, XAI ve Değerlendirme Stratejisi

## 1. Model ailesi seçimi

### PCA

**Neden ilk model?**

- Hızlı, deterministik ve az hiperparametreli.
- Çok değişkenli kanal ilişkilerini doğrusal alt uzayda temsil eder.
- Reconstruction residual doğrudan kanal katkısı verir.
- Veri, scaling ve threshold hatalarını karmaşık model arkasına gizlemez.

Ana skor:

```text
x_hat = inverse_transform(transform(x))
e_j = (x_j - x_hat_j)^2
score = mean(e_j)
```

PCA açıklaması residual katkısı ve gerekirse principal component loading'leri
ile yapılır.

### Isolation Forest

Pencere özetleri veya latent feature'lar üzerinde dağılımdan ayrışan örnekleri
yakalamak için kullanılır. Zaman sırasını kendiliğinden modellemediğinden ham
sequence yerine istatistiksel pencere özellikleriyle başlanır. Feature
contribution doğal değildir; SHAP veya perturbation gerekir.

### One-Class SVM

Küçük/orta veri ve iyi ölçeklenmiş feature uzayında karşılaştırma baseline'ıdır.
Yüksek boyut, büyük örnek sayısı ve `nu`/kernel hassasiyeti nedeniyle ana üretim
adayı olarak görülmez.

### Dense Autoencoder

PCA'nın doğrusal olmayan karşılığıdır. İlk derin model olmalıdır çünkü temporal
mimari eklemeden kanal ilişkilerinin katkısını test eder. Kapasite kontrollü,
dar bottleneck ve regularization gerekir.

### TCN Autoencoder

Dilated causal convolution ile farklı zaman ölçeklerini paralel ve genellikle
kararlı biçimde modeller. Uydu telemetrisindeki yerel trend, periyodik desen ve
gecikmeli etkiler için güçlü ilk temporal adaydır.

### LSTM Autoencoder

Uzun bağımlılıkları modellemek için TCN ile aynı protokolde karşılaştırılır.
Yalnızca daha tanınmış olduğu için seçilmez; event recall, delay, false alarm ve
maliyet sonuçlarına göre karar verilir.

### Transformer

Uzun pencere, yüksek veri hacmi ve karmaşık cross-channel bağımlılık varsa
opsiyoneldir. Küçük veri, seyrek anomaly ve açıklama maliyeti nedeniyle MVP
kapsamında değildir.

### GNN / GAT

Grafiğin anlamlı bir kaynağı varsa kullanılır:

- Elektrik/termal/fonksiyonel mimari.
- Aynı subsystem üyeliği.
- Bilinen sinyal akışı veya gecikmeli fiziksel etki.
- Yalnız training nominal verisinden çıkarılmış kararlı edge'ler.

Salt tam-veri korelasyon grafiği leakage ve sahte nedensellik riski taşır.

## 2. Üç seviyeli XAI

### A. Kanal bazlı açıklama

Yanıtlanan soru: "Alarm skorunu en çok hangi telemetri kanalları taşıdı?"

#### Reconstruction error attribution

PCA ve autoencoder için doğal başlangıçtır:

```text
raw_contribution_j = sum_t mask[t] * (x[t,j] - x_hat[t,j])²
normalized_contribution_j =
    raw_contribution_j / sum_k(raw_contribution_k)
```

Validation nominal kanal hata dağılımı farklıysa önce robust standardizasyon:

```text
z_error_j = (error_j - median_nominal_j) / (IQR_nominal_j + epsilon)
```

Sonuç top-k sıralama, katkı yüzdesi ve sapma yönüyle sunulur.

#### SHAP

- Isolation Forest veya tabular pencere modeli için uygundur.
- Background set yalnızca nominal training örneklerinden seçilir.
- Çok korelasyonlu kanallarda katkının kanallar arasında paylaşılabileceği
  raporda belirtilir.
- Hesap maliyeti nedeniyle event çevresinde örnekleme yapılır.

#### Integrated Gradients

- PyTorch temporal modellerinde score'un input sequence'e gradient katkısını
  verir.
- Baseline; sıfır vektör değil, nominal rejim medyanı veya bağlama uygun
  referans pencere olmalıdır.
- Birden fazla baseline ile duyarlılık analizi yapılır.

#### Occlusion

Bir kanal veya zaman bloğu nominal referansla değiştirilir:

```text
importance = original_score - occluded_score
```

Model bağımsız ve anlaşılırdır. Hesap maliyeti yüksektir ama ilk kritik event
raporlarında doğrulama yöntemi olarak değerlidir.

### B. Zaman bazlı açıklama

Yanıtlanan sorular:

- Sapma ilk ne zaman başladı?
- Skoru en çok hangi alt pencere yükseltti?
- Alarm ne kadar gecikti?

Yöntem:

1. Kanal-zaman attribution matrisi üret.
2. Kanal boyunca toplayarak temporal importance çıkar.
3. Validation tabanlı kritik importance threshold'u uygula.
4. Bitişik kritik adımları interval'e birleştir.
5. En yüksek toplam katkılı interval'i `critical_window` seç.

Zaman tanımları:

- `event_start`: ground truth anomaly başlangıcı.
- `first_detection`: filtrelenmiş ilk alarm.
- `peak_time`: event içindeki en yüksek score.
- `detection_delay = first_detection - event_start`.
- `early_warning_time = failure_or_critical_time - first_detection`.

Negatif detection delay, label başlangıcından önce uyarı anlamına gelir; false
positive olup olmadığı operasyon/label toleransına göre ayrıca incelenir.

### C. Mühendislik ve subsystem açıklaması

Kanal kataloğu görev bazında tutulur:

| Alan | Örnek |
|---|---|
| `channel_id` | `battery_voltage` |
| `display_name` | Battery Bus Voltage |
| `unit` | V |
| `subsystem` | EPS |
| `component` | Battery |
| `nominal_range` | Görev özel |
| `related_channels` | current, temperature, SOC |
| `valid_modes` | sunlit, eclipse, safe |

Subsystem katkısı kanal katkılarının doğrudan toplamı yerine kanal sayısına
göre normalize edilmelidir; aksi halde çok kanallı subsystem haksız avantaj
elde eder.

Örnek mühendislik çevirisi:

```text
Model bulgusu:
battery_voltage residual yüksek, battery_current ters yönde artıyor,
eclipse context aktif.

Mühendislik ifadesi:
EPS grubunda eclipse sırasında beklenen gerilim-akım ilişkisi bozuldu.
Batarya sıcaklığı ve state-of-charge telemetrisiyle birlikte incelenmeli.
```

Bu ifade "batarya arızalıdır" demez. SAK olası subsystem ve inceleme önceliği
önerir; nedensel teşhis iddiası uzman doğrulaması gerektirir.

## 3. Erken uyarı mantığı

### Threshold

- Sabit quantile: validation nominal skorunun örneğin `%99.5` yüzdeliği.
- Robust: median + `k * MAD`.
- EVT: yeterli kuyruk örneği varsa opsiyonel.
- Mod bazlı: safe, eclipse, payload-active gibi rejimler için ayrı threshold.

Threshold yalnız validation/calibration verisiyle seçilir.

SAK sentetik deneyinde global threshold'a ek olarak `operational_mode` bazlı
kalibrasyon da raporlanır. Validation nominal skorları önce aynı EWMA ile
smooth edilir; sonra her mode için ayrı quantile threshold üretilir. Test
aşamasında her timestamp kendi operational mode threshold'u ile karşılaştırılır.
Validation'da görülmeyen veya minimum örnek sayısına ulaşmayan mode değeri
global threshold'a düşer. Bu yaklaşım safe, nominal ve payload rejimlerinin score
dağılımları farklı olduğunda yanlış alarm/noise dengesini daha okunur yapar.

### Dynamic threshold

Score dağılımı operasyon bağlamıyla değişiyorsa causal rolling median/MAD veya
mode-conditioned threshold kullanılabilir. Dinamik eşik anomaly dönemini
hızla normal kabul etmemeli; referans güncellemesi alarm sırasında
dondurulmalıdır.

### Smoothing ve persistence

- EWMA kısa spike'ları bastırır.
- Causal moving average yorumlanabilir bir alternatiftir.
- `m-of-n`: son `n` adımın en az `m` tanesi threshold üstündeyse alarm.
- Cooldown aynı olayı tekrar tekrar raporlamayı önler.

### Event-aware filtering

- Bitişik alarm noktalarını tek event yap.
- Kısa boşlukları merge tolerance ile birleştir.
- Minimum event süresi uygula.
- Known maneuver/maintenance interval'larını suppress etmek yerine ayrı
  `expected_context` etiketiyle raporla.
- Aynı subsystem ve kısa zaman aralığındaki alarmları ilişkilendir.

### Risk seviyesi

Risk yalnız score büyüklüğü değildir:

```text
risk = f(score_excess, persistence, affected_channels,
         subsystem_criticality, uncertainty, context)
```

İlk sürüm kural tabanlı olabilir:

- Low: threshold üstü ancak kısa/tekil.
- Medium: kalıcı, birden çok ilişkili kanal.
- High: kritik subsystem, yüksek excess ve düşük belirsizlik.

## 4. Değerlendirme metrikleri

### Point-wise precision, recall, F1

Her timestamp ayrı sınıflandırılır. Uzun anomaly event'leri metriği
şişirebildiği ve birkaç adımlık kaymayı ağır cezalandırdığı için tek başına
kullanılmaz.

### Event-wise precision, recall, F1

Tahmin event'i ground truth interval ile örtüşüyorsa veya tanımlı tolerans
içinde başlıyorsa eşleşir. One-to-one matching uygulanır; bir gerçek olaya
çok sayıda alarm göndermek tek başarı sayılmaz ve fazladan alarmlar false
positive olur.

### Detection delay

```text
delay = first_matched_alarm_time - true_event_start
```

Ortalama, medyan, p90 ve missed event oranıyla raporlanır.

### False alarms per day/orbit

Operasyon açısından precision'dan daha anlaşılır olabilir:

```text
false_alarm_rate = unmatched_predicted_events / observed_days_or_orbits
```

### Early warning time

Etiketlerde failure/critical point varsa:

```text
early_warning_time = critical_time - first_valid_alarm
```

Pozitif değer kullanılabilir müdahale süresini gösterir.

### Anomaly score stability

- Aynı nominal rejimde score varyansı.
- Seed'ler arası score korelasyonu.
- Küçük input perturbation altında sıralama ve threshold crossing değişimi.

### Explanation consistency

- `Hit@k`: injected/known affected channel top-k içinde mi?
- nDCG: kanal sıralamasının ground truth sırasına uyumu.
- Rank correlation: seed/model tekrarlarında attribution sırası.
- Infidelity: önemli kanal occlusion'ı score'u beklenen miktarda değiştiriyor mu?
- Subsystem accuracy/Hit@k: doğru subsystem aday listesinde mi?

### Ana model seçme tablosu

Her deney şu sütunları raporlar:

```text
event_precision
event_recall
event_f1
median_detection_delay
p90_detection_delay
false_alarms_per_day_orbit
channel_hit_at_3
explanation_stability
inference_latency
model_size
```

Tek bir birleşik accuracy yerine Pareto değerlendirmesi yapılır. Model seçimi
önce false alarm bütçesi, sonra event recall ve delay, ardından açıklama
kalitesi ve maliyet üzerinden yapılır.

## 5. Sentetik veri ve anomaly injection

### Nominal simülatör

Simülatör en az şu yapıları içermelidir:

- Yörünge fazına bağlı periyodik sıcaklık.
- Eclipse sırasında solar array akımının düşmesi.
- Batarya gerilim, akım ve SOC ilişkisi.
- Thermal inertia ile gecikmeli sıcaklık tepkisi.
- Operasyon moduna bağlı payload ve güç tüketimi.
- Kanal başına farklı sampling, gürültü ve kısa veri boşlukları.
- Ortak latent etkenlerle ilişkili kanal grupları.

Basit denklem örneği:

```text
sunlight(t) = square_wave(orbit_phase)
solar_current(t) = max(0, base * sunlight(t)) + noise
load_current(t) = mode_load(mode(t)) + noise
battery_current(t) = load_current(t) - solar_current(t)
temperature(t) = thermal_lag(sunlight, load_current) + noise
```

### Injection tipleri

| Tip | Üretim | Beklenen XAI |
|---|---|---|
| Spike | Tek/kısa süreli yüksek genlik | Enjekte edilen kanal, dar zaman |
| Drift | Zamana bağlı lineer/eğrisel offset | Kanal ve drift başlangıcı |
| Step change | Sabit ani offset | Kanal, değişim noktası |
| Slow degradation | Eğimi giderek artan sapma | Erken zaman katkısı artışı |
| Stuck sensor | Değeri sabitleme | Kanal, düşük varyans interval'i |
| Noise increase | Gürültü varyansını yükseltme | Kanal ve yaygın zaman katkısı |
| Correlation break | Bir kanalı ortak latent ilişkiden ayırma | İlişkili kanal grubu |
| Thermal runaway | Pozitif eğimli/ivmeli sıcaklık | Thermal kanalları |
| Voltage drop + current rise | İki kanala ters yönlü değişim | EPS ve iki ana kanal |
| Orbit-dependent thermal | Yalnız belirli orbit fazında offset | Thermal + orbit context |

Her injection manifesti şu alanları taşır:

```yaml
event_id: SYN-0001
type: voltage_drop_current_rise
start: 2026-01-01T12:00:00Z
end: 2026-01-01T12:30:00Z
affected_channels:
  - battery_voltage
  - battery_current
expected_subsystem: EPS
parameters:
  voltage_delta: -1.5
  current_delta: 0.8
```

### XAI doğrulaması

1. Injection bilinmeden model eğitilir.
2. Alarm ve explanation otomatik üretilir.
3. Top-k kanal, interval ve subsystem manifestle karşılaştırılır.
4. Hit@k, temporal IoU ve subsystem Hit@k hesaplanır.
5. Attribution dışındaki etkilenmiş kanallar, anomaly propagation olarak ayrıca
   incelenir.

Bu yöntem gerçek anomaly kök nedeni etiketlerinin az olduğu durumda XAI
pipeline'ını test eder; gerçek uzman doğrulamasının yerini tutmaz.
