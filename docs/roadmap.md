# SAK Aşamalı Geliştirme Yol Haritası

Her sürüm bir araştırma sorusunu kapatmalı ve sonraki sürüme ölçülebilir bir
giriş kriteri sağlamalıdır. Yalnızca kodun tamamlanması sürüm başarısı sayılmaz.

## SAK-v0 — Veri altyapısı ve sentetik test

**Amaç**

Dataset bağımsız, leakage kontrollü ve test edilebilir telemetri pipeline'ı
kurmak.

**Yöntem**

- Kanonik timestamp-indexed veri şeması.
- CSV/Parquet adaptörü ve kaynak manifesti.
- Resampling, eksik veri maskesi ve train-only normalization.
- Kronolojik train/validation/test split.
- Window/sequence builder.
- Fiziksel ilişkili temel sentetik telemetri üretimi.
- Parametreli anomaly injection ve injection ground truth'u.
- Telemetri, context ve label görselleri.

**Beklenen çıktı**

- Bir komut/config ile sentetik dataset üretimi.
- Deterministik preprocessing artefact'ı.
- Window metadata ve leakage testleri.
- Veri kalite raporu.

**Başarı kriteri**

- Aynı seed aynı veriyi ve split'i üretir.
- Hiçbir window split sınırını geçmez.
- Scaler istatistikleri yalnızca train verisinden gelir.
- Injection interval ve kanalları machine-readable manifestte bulunur.
- Eksik/duplicate timestamp testleri geçer.

**Riskler**

- Sentetik verinin fazla kolay ve gerçek dışı olması.
- Interpolation'ın anomaly biçimini bozması.
- Operasyon modu değişimlerinin yanlışlıkla anomaly etiketi alması.

**SAK-v1'e geçiş kriteri**

En az üç nominal rejim, beş anomaly tipi ve bilinen kanal katkılarıyla uçtan
uca veri pipeline'ının testlerden geçmesi.

## SAK-v1 — PCA ve istatistiksel baseline

**Amaç**

Basit, hızlı ve açıklanabilir referans performansı oluşturmak; skor, threshold
ve event evaluation protokolünü sabitlemek.

**Yöntem**

- Sabit limit/rate-of-change kontrolleri.
- Rolling z-score, moving average ve EWMA residual.
- PCA reconstruction error.
- Isolation Forest karşılaştırması.
- Opsiyonel One-Class SVM küçük/orta veri üzerinde.
- Validation nominal quantile veya robust threshold.
- Kanal bazlı standardize residual katkısı.

**Beklenen çıktı**

- Model karşılaştırma tablosu.
- Ham ve filtrelenmiş score zaman çizgisi.
- Top-k kanal listesi.
- İlk SAK Early Warning Report.
- Point-wise ve event-wise metrikler.

**Başarı kriteri**

- Sentetik event'lerin çoğu anlamlı event overlap ile yakalanır.
- Tek noktalı spike dışındaki anomalilerde detection delay raporlanır.
- False alarms/day veya orbit ölçülür.
- PCA açıklamalarında enjekte edilen ana kanal top-k içinde yer alır.
- Threshold test setine bakmadan seçilir.

**Riskler**

- PCA doğrusal olmayan davranışı kaçırabilir.
- Birden fazla operasyon rejimi reconstruction error'u büyütebilir.
- Isolation Forest zaman sırasını doğrudan kullanmaz.
- OCSVM kanal sayısıyla pahalı ve hiperparametreye hassastır.

**SAK-v2'ye geçiş kriteri**

Baseline değerlendirme pipeline'ı dondurulmuş, en az bir baseline kabul
edilebilir false alarm bütçesinde çalışıyor ve açıklama doğruluğu ölçülebiliyor
olmalıdır.

## SAK-v2 — Autoencoder tabanlı anomaly detection

**Amaç**

Doğrusal olmayan kanal ilişkileri ve zamansal davranış için baseline üstü
katkıyı ölçmek.

**Yöntem**

- Dense Autoencoder.
- TCN Autoencoder ve LSTM Autoencoder karşılaştırması.
- Reconstruction error tabanlı aggregate ve channel-time score.
- Early stopping, checkpointing ve seed tekrarı.
- Operasyon moduna koşullandırma veya mode-specific modeller.
- Kanal-zaman heatmap'i.

**Beklenen çıktı**

- PCA ve AE ailesinin aynı test protokolünde karşılaştırması.
- Event-level performans ve alarm gecikmesi.
- Kanal-zaman reconstruction heatmap'leri.
- Model boyutu, inference süresi ve bellek ölçümü.

**Başarı kriteri**

- En az üç seed sonucuyla ortalama ve varyans raporu.
- Baseline'a göre event recall veya false alarm bütçesinde anlamlı iyileşme.
- Açıklama top-k başarısının baseline'dan kötüleşmemesi.
- Causal inference ve tekrar üretilebilir checkpoint.

**Riskler**

- AE'nin anomaly içeren eğitim verisini de reconstruct etmeyi öğrenmesi.
- Aşırı kapasite nedeniyle anomalileri düşük hatayla reconstruct etmesi.
- LSTM eğitim kararsızlığı ve uzun eğitim süresi.
- Reconstruction error'un kök neden yerine etkilenen kanalı göstermesi.

**SAK-v3'e geçiş kriteri**

En az bir derin model baseline'a karşı ölçülebilir değer sağlamalı; sağlamıyorsa
SAK-v3 baseline/PCA üzerinden devam eder. Derin model kullanımı zorunlu değildir.

## SAK-v3 — Explainable AI ve mühendislik raporu

**Amaç**

Model kararını tutarlı, test edilebilir ve mühendis tarafından yorumlanabilir
bir açıklamaya dönüştürmek.

**Yöntem**

- Reconstruction error attribution.
- Temporal/channel occlusion.
- PyTorch modellerinde Integrated Gradients.
- Uygun modellerde SHAP.
- Kanal-subsystem bilgi tabanı.
- Context-aware şablon ve kural tabanlı mühendislik yorumları.
- Attribution uncertainty ve seed/model tutarlılığı.

**Beklenen çıktı**

- Top-k kanal ve kritik zaman aralığı.
- Olası subsystem sıralaması.
- Açıklama heatmap'i.
- Confidence/uncertainty alanı.
- Markdown/JSON SAK Early Warning Report.

**Başarı kriteri**

- Sentetik ground truth kanalları için Hit@k ve nDCG raporu.
- Benzer girdilerde explanation stability ölçümü.
- Occlusion ile seçilen kanallar kaldırıldığında skorun beklenen yönde düşmesi.
- Örnek olayların uzman veya domain danışmanı tarafından yararlı bulunması.
- Rapor, yalnız model skoruna dayanarak kesin kök neden iddia etmez.

**Riskler**

- SHAP/IG sonuçlarının baseline ve feature ölçeğine hassas olması.
- Attribution'ın korelasyon nedeniyle neden-sonuç gibi yorumlanması.
- Subsystem mapping'in görevden göreve değişmesi.
- Kullanıcıya aşırı kesinlik hissi verilmesi.

**SAK-v4'e geçiş kriteri**

Kanal ilişkileri için güvenilir fiziksel bilgi veya kararlı istatistiksel yapı,
güçlü event-level baseline ve doğrulanmış XAI değerlendirme protokolü bulunmalı.

## SAK-v4 — GNN / Graph Attention

**Amaç**

Kanal ilişkilerindeki bozulmayı ve alt sistemler arası anomaly yayılımını graph
üzerinde modelleyerek kök neden sıralamasını geliştirmek.

**Yöntem**

- Node: telemetri kanalı.
- Node feature: son pencere değerleri, residual ve context.
- Edge: fiziksel bağlantı, subsystem bilgisi veya train-only kararlı ilişki.
- Graph autoencoder, temporal GNN veya GAT.
- Node/edge attribution ve anomalous subgraph.
- Physics-informed ve correlation-only graph ablation.

**Beklenen çıktı**

- Graph anomaly score.
- Node ve edge önem sıralaması.
- Anomaly propagation görünümü.
- Kök neden adayı ve etkilenmiş kanal ayrımı.

**Başarı kriteri**

- PCA/TCN baseline'a karşı event veya root-cause metriğinde anlamlı iyileşme.
- Graph seçiminin ablation testleriyle gerekçelendirilmesi.
- Node importance'ın sentetik correlation-break ground truth'uyla uyumu.
- Ek hesap maliyetinin raporlanması.

**Riskler**

- Korelasyonun fiziksel nedensellik sanılması.
- Operasyon moduyla değişen edge'lerin sabit kabul edilmesi.
- Küçük veri üzerinde aşırı parametre ve overfitting.
- GNN açıklamalarının kararsız olması.

**Tamamlanma kriteri**

GNN'nin yalnızca daha karmaşık değil, false alarm, event recall, gecikme veya
kök neden kalitesinde baseline üstü ölçülebilir değer sağladığı gösterilmelidir.
Aksi durumda üretim adayı daha basit model olarak kalır.

## Model sıralaması kararı

```text
Data quality
  -> statistical controls
  -> PCA
  -> Isolation Forest / OCSVM
  -> Dense AE
  -> TCN and LSTM AE
  -> optional Transformer
  -> conditional GNN/GAT
```

GNN'ye erken geçilmez. Önce veri sözleşmesi, skor semantiği, threshold,
event-level değerlendirme, false alarm bütçesi ve XAI raporu oturmalıdır.

