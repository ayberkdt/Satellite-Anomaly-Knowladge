# Araştırma ve Rapor Planı

## 1. Ana araştırma sorusu

Çok değişkenli uydu telemetrisinde nominal davranıştan sapmaları düşük yanlış
alarm ve kabul edilebilir gecikmeyle tespit ederken, alarmı kanal, zaman ve alt
sistem düzeyinde mühendislik açısından açıklamak mümkün müdür?

## 2. Alt araştırma soruları

1. PCA ve istatistiksel baseline'lar gerçek/sentetik event'lerde ne kadar güçlü?
2. Dense veya temporal autoencoder event recall ve delay'i iyileştiriyor mu?
3. EWMA ve persistence yanlış alarmı hangi gecikme maliyetiyle azaltıyor?
4. Reconstruction attribution gerçek etkilenen kanalları top-k içinde buluyor mu?
5. Integrated Gradients/occlusion doğal reconstruction açıklamasına ek değer
   sağlıyor mu?
6. Operasyon modu, eclipse ve orbit context'i false alarmı azaltıyor mu?
7. Yeterli veri ve fiziksel graph varsa GNN kök neden sıralamasını geliştiriyor mu?

## 3. Hipotezler

- H1: Event-aware filtre, küçük recall kaybı karşılığında false alarm/event
  oranını anlamlı biçimde düşürür.
- H2: Temporal AE, yavaş drift ve ilişki bozulmalarında PCA'dan daha erken
  tespit sağlar.
- H3: Context-aware threshold, mode geçişlerinde false alarmı azaltır.
- H4: Reconstruction contribution, kontrollü injection'larda etkilenen kanalı
  top-3 içinde yüksek oranda bulur.
- H5: Fiziksel graph, correlation-only graph'tan daha kararlı açıklama üretir.

## 4. Deney tasarımı

### Datasetler

1. SAK sentetik telemetri: kontrollü anomaly ve açıklama ground truth'u.
2. NASA/JPL SMAP-MSL: literatür karşılaştırması ve anomaly interval'ları.
3. ESA Anomaly Dataset: çok görevli, çok alt sistemli gerçek telemetri.
4. Erişim sağlanırsa görev-özel TUSAŞ/sanayi verisi.

Datasetler birbirine karıştırılmadan ayrı protokollerle raporlanır. SMAP/MSL
verisi kanal bazlı seriler ve ESA verisi daha zengin görev bağlamı sunabileceği
için doğrudan tek bir skor tablosunda karşılaştırılmayabilir.

### Split

- Kronolojik train/validation/test.
- Anomaly event bütünlüğü korunur.
- Train nominal-only ve robust mixed-train ayrı deneylerdir.
- Threshold validation/calibration partition ile seçilir.
- Test model seçimi için kullanılmaz.

### Karşılaştırmalar

- Statistical vs PCA vs Isolation Forest.
- PCA vs Dense AE.
- TCN AE vs LSTM AE.
- Global threshold vs mode-conditioned threshold.
- Raw threshold vs EWMA vs EWMA+persistence.
- Reconstruction attribution vs IG vs occlusion.
- İleri aşamada physical graph vs correlation graph vs no-graph.

### Ablation

- Context olmadan.
- Persistence olmadan.
- Channel error normalization olmadan.
- Imputation mask olmadan.
- Subsystem normalization olmadan.
- GNN'de belirli edge tipleri olmadan.

## 5. Sonuç sunumu

Minimum tablo ve görseller:

- Dataset kanal, süre, missingness ve event istatistikleri.
- Model event precision/recall/F1 tablosu.
- False alarms/day-orbit ve detection delay tablosu.
- Score ve threshold zaman çizgisi.
- Kanal-zaman attribution heatmap'i.
- Channel Hit@k ve explanation stability.
- Hiperparametre/threshold sensitivity.
- Başarılı ve başarısız olay incelemeleri.
- Hesap maliyeti ve inference latency.

Başarısız vakalar saklanmalı ve tartışılmalıdır. Yüksek accuracy, özellikle
uzun nominal periyotlarda, tek başına anlamlı sonuç değildir.

## 6. Önerilen rapor yapısı

### Introduction

Operasyonel telemetri hacmi, erken tespit ihtiyacı, yanlış alarm maliyeti ve
açıklanabilirlik problemi.

### Problem Definition

Nominal davranış öğrenimi, anomaly event tanımı, erken uyarı ve mühendislik
açıklamasının kapsamı.

### Satellite Telemetry Anomaly Detection

Noktasal, bağlamsal ve toplu anomaliler; seyrek etiket, yüksek boyut, farklı
sampling ve mode değişimi zorlukları.

### Explainable AI for Space Operations

Feature attribution ile mühendislik yorumu arasındaki fark, güven ve insan
döngüsü.

### Dataset and Preprocessing

Kaynaklar, şema, resampling, missingness, split, context ve leakage önlemleri.

### Methodology

Pipeline, ortak model sözleşmesi, score kalibrasyonu ve event builder.

### Baseline Models

İstatistiksel kontroller, PCA, Isolation Forest ve OCSVM.

### Autoencoder Models

Dense AE, TCN/LSTM AE mimarisi, loss ve eğitim protokolü.

### Explainability Layer

Reconstruction attribution, IG/SHAP/occlusion, subsystem mapping.

### Early Warning Logic

Threshold, EWMA, persistence, event merge, risk level.

### Experiments

Araştırma soruları, split, hiperparametreler, seed'ler ve ablation.

### Results

Event, delay, false alarm, XAI ve maliyet sonuçları.

### Discussion

Operasyonel anlam, model karmaşıklığının getirisi, domain farkı.

### Limitations

Etiket kalitesi, sentetik-gerçek farkı, causal interpretation sınırı, veri
gizliliği ve görev bağımlılığı.

### Future Work

Mode-aware online learning, uncertainty, transfer learning, GNN/GAT ve uzman
geri besleme döngüsü.

## 7. Kod ve deney standartları

- Python 3.11+, PyTorch, scikit-learn, pandas ve NumPy.
- Config-driven deney; hard-coded path ve hiperparametre yok.
- Global ve data-loader seed kontrolü.
- Model, optimizer, scaler, kanal sırası ve config checkpoint'i.
- Structured logging.
- MLflow veya eşdeğer experiment tracking opsiyonu.
- Type hint ve kısa, davranış odaklı docstring.
- Birim test, integration test ve leakage regression test.
- Notebook yalnız EDA/görselleştirme; ana kod `src/sak` içinde.
- En az üç seed ve ortalama/standart sapma.

## 8. Kritik kısıtlar

- Yüksek accuracy tek başına yeterli değildir.
- False alarm oranı operasyonel kabulün ana belirleyicisidir.
- Açıklanabilirlik yalnız heatmap değil, sınırlılıkları belirtilmiş mühendislik
  yorumudur.
- GNN ileri aşamadır ve baseline üstü katkısını kanıtlamalıdır.
- XAI ilk modelden itibaren ortak çıktı sözleşmesindedir.
- Model bulguları fiziksel bağlam ve kanal ilişkileriyle kontrol edilir.
- Gerçek uydu verisi az etiketli, etiketsiz veya label sınırları belirsiz olabilir.
- Operasyon modu değişimleri yanlış anomaly olarak görülebilir.
- Sistem alarm gecikmesini ve erken uyarı süresini ölçer.
- Sistem öneri üretir; uçuş-kritik nihai karar vermez.

## 9. Doğrulanmış başlangıç kaynakları

- [ESA Anomaly Dataset, Zenodo, DOI 10.5281/zenodo.12528696](https://zenodo.org/records/12528696)
- [ESA dataset executive summary](https://nebula.esa.int/sites/default/files/neb_tec_studies/3296/public/ESA-AI-ADSB-RP-000008%20ESR_ExecutiveSummaryReport.pdf)
- [NASA/JPL Telemanom repository and SMAP-MSL metadata](https://github.com/khundman/telemanom)
- [ESA telemetry and telecommand overview](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Onboard_Computers_and_Data_Handling/Telemetry_Telecommand)
- [ESA few-shot anomaly detection activity](https://activities.esa.int/4000141301)

ESA Anomaly Dataset, 25 Haziran 2024 tarihli kaydında üç gerçek görevden
yaklaşık 12 GB sıkıştırılmış veri sunar. ESA yürütücü özetinde 9 yıllık veri,
224 telemetri kanalı, 12 alt sistem ve 1006 rare/anomaly event belirtilir.
Bu büyüklükler indirme ve preprocessing planında dikkate alınmalıdır.

