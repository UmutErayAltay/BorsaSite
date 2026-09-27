# Backtest Audit — Faz 0

Tarih: 2026-09-25. Kapsam: `pipeline/`, `trading/`, `config/`, `scripts/`, `api/`,
`tests/`, `.github/workflows/`, DB şeması (`pipeline/db.py::SCHEMA_SQL`), README.
Hiçbir kod değiştirilmedi — bu doküman sadece mevcut durumun tespitidir.

## 0. Sistemin bugünkü şekli (özet)

Günlük pipeline (`pipeline/daily_pipeline.py`, GitHub Actions'ta hafta içi
16:00 UTC = 19:00 İstanbul'da, BIST kapanışından sonra çalışıyor):

```
fetch_prices → fetch_news (RSS) → kap_sync → relink_all_news + rebuild_sentiment_daily
  → analyze_sentiment (FinBERT/TR-BERT) → train_model (HER GÜN, GH Actions kalıcı disk
  tutmadığı için) → predict_model (sadece sembol başına en güncel satır) → trading.engine.run_once
```

Model: XGBoost, 12 feature (RSI14, MACD/sinyal/hist, close/SMA20 ve SMA50 oranı,
return_1d, return_5d, volume_ratio, sentiment_avg_3d, sentiment_news_3d, is_bist).
Target: `target_up = close.shift(-1) > close` (yarın kapanış > bugün kapanış).
Trading: eşik-bazlı (`buy_threshold=0.55`, `sell_threshold=0.50`), sadece BIST,
maks 8 açık pozisyon, pozisyon başı bakiyenin %15'i, toplam risk %90, maks 10 gün tutma.

## 1. Güçlü taraflar

- Veri katmanı (fiyat/haber/KAP/sentiment) olgun ve gerçek: 8 ay boyunca kullanılmış,
  4 bağımsız entity-linking hatası bulunup düzeltilmiş (bkz. commit `592b457`),
  gerçek üretim olayları (Supabase pooler prepared-statement hatası, stored XSS,
  crash-loop) tespit edilip giderilmiş.
- Finansal mantık (`trading/costs.py`, `trading/portfolio.py`) SAF fonksiyonlar
  olarak izole edilmiş, iyi test edilmiş (portfolio: 9 test, engine: 6 test, costs: 4 test).
- Her karar (al/sat/tut/red) `trade_decisions` tablosuna insan-okunur gerekçeyle
  loglanıyor — "neden bu işlem reddedildi" sorusu zaten cevaplanabilir durumda.
- `portfolio_snapshots` günlük equity kaydı zaten var — bir equity curve'ün ham
  verisi hazır, sadece raporlanmıyor.
- Config zaten kısmen ayrıştırılmış (`config/trading.yaml`, `config/model.yaml`).
- Gerçek emir gönderme kodu hiç yazılmamış — LIVE trading riski yapısal olarak yok.

## 2. Kritik bulgu: son günün target'ı YANLIŞ etiketleniyor (doğrulandı)

`pipeline/features.py:65`:
```python
out["target_up"] = (close.shift(-1) > close).astype(float)
```
`close.shift(-1)` bilinmeyen (henüz gerçekleşmemiş) son gün için `NaN` döner.
Pandas'ta `NaN > x` karşılaştırması `NaN` değil **`False`** döner — `.astype(float)`
bunu **`0.0`** yapar. Doğrulandı (gerçek pandas ile):
```python
>>> (pd.Series([10,11,9,12]).shift(-1) > pd.Series([10,11,9,12])).astype(float).tolist()
[1.0, 0.0, 1.0, 0.0]   # son eleman 0.0, NaN DEĞİL
```
`dataset.py:111`'deki `dropna(subset=[...,"target_up"])` bu satırı YAKALAMIYOR
çünkü değer `0.0`, `NaN` değil.

**Sonuç:** Her pipeline çalışmasında, en güncel işlem gününün satırı
"düştü" (`target_up=0`) diye **yanlış** etiketlenip modele **eğitim verisi**
olarak veriliyor — gerçekte o günün gerçek sonucu henüz bilinmiyor. Aynı satır,
`predict_model.py`'de `.tail(1)` ile **canlı tahmin** üretmek için de kullanılıyor
(feature'lar target'ı içermediği için tahminin kendisi bozulmuyor, ama o gün
modelin "gördüğü" örnek kümesinde sistematik bir yanlış etiket var). Sembol
başına günde 1 satır olduğu için etkisi büyük ihtimalle küçük ama **gerçek ve
sessiz** bir veri kalitesi hatası — düzeltilmeden walk-forward/backtest
sonuçları da bu hatayı miras alır.

Düzeltme yönü (uygulanmadı, sadece tespit): son günü `target_up` hesaplanmadan
önce ayrı tutup `NaT`/`NaN` ile işaretlemek, ya da `close.shift(-1).notna()`
maskesiyle açıkça filtrelemek.

## 3. Look-ahead / veri sızıntısı riskleri

| Risk | Durum | Detay |
|---|---|---|
| Teknik göstergeler (RSI/MACD/SMA/return/volume) | **Sızıntı yok** | Sadece `date <= D` fiyatlarını kullanıyor, target ayrı sütun. |
| `sentiment_avg_3d` / `sentiment_news_3d` | **Sızıntı yok, ama kırılgan** | `rolling(lag=3, min_periods=1)` **sağa hizalı** (D, D-1, D-2 ortalaması) — D gününün kendi sentiment'ını içeriyor. Bu doğru bir varsayım SADECE eğer D günü haberlerinin TAMAMI D'nin kapanışına kadar toplanmışsa (canlıda öyle: pipeline kapanıştan sonra 19:00'da çalışıyor). Ama bu varsayım kodda hiçbir yerde açıkça doğrulanmıyor/test edilmiyor. |
| `rebuild_sentiment_daily()` tarih ataması | **Küçük ama gerçek zaman dilimi hatası** | `date(COALESCE(n.published_at, n.fetched_at))` — `published_at` UTC olarak saklanıyor (`fetch_news_rss.py::parse_published`), İstanbul'a çevrilmeden doğrudan `date()`'e kesiliyor. 21:00–24:00 UTC (00:00–03:00 İstanbul) arası yayınlanan bir haber, İstanbul takviminde ERTESİ gün iken UTC takviminde BUGÜN görünür — nadir ama sistematik bir off-by-one-day riski. |
| Son gün etiket hatası | **Gerçek bug, bkz. §2** | Yukarıda. |
| `sentiment_lag_days` isimlendirmesi | **Yanıltıcı** | "lag" deniyor ama gerçekte lag YOK — trailing window D'yi de içeriyor. Gerçek bir lag (D-1..D-3) istenseydi `.shift(1).rolling(...)` gerekirdi. Şu anki hâli yanlış değil (yukarıdaki varsayım doğruysa) ama isim yanıltıcı, gelecekte biri "lag" sanıp yanlış varsayımla kod değiştirebilir. |
| Model eğitimi | **Sızıntı yok ama optimist** | `train_model.py::_temporal_split` tarihe göre sıralı tek bölme (%80/%20) — geleceği training'e sızdırmıyor, ama TEK bölme; hangi tarih aralığına denk geldiği şansa bağlı, walk-forward yok. |

## 4. Backtest eksikleri (en büyük boşluk)

- **Gerçek bir historical backtest motoru yok.** `trading/engine.py` yalnızca
  CANLI paper-trading mantığı: her çağrıda `predictions` ve `prices_daily`
  tablosundaki **en güncel** satırı okuyor (`ORDER BY ... DESC LIMIT 1`).
  Geçmiş bir tarih aralığını "o tarihte ne bilinebilirdi" kısıtıyla yeniden
  oynatmak mümkün değil.
- Walk-forward validation yok — tek train/test bölmesi.
- Equity curve, drawdown, Sharpe/Sortino, profit factor, expectancy gibi
  strateji metrikleri hiç hesaplanmıyor/raporlanmıyor (ham veri
  `portfolio_snapshots`'ta var ama işlenmiyor).
- Benchmark (buy&hold, eşit ağırlıklı BIST sepeti) karşılaştırması yok.
- Threshold seçimi (`buy_threshold=0.55`) hiçbir validasyon sürecinden
  geçmeden config'e sabit yazılmış — hangi veriyle seçildiği belgelenmemiş.
- Probability calibration (Brier score, reliability diagram) yok.

## 5. İşlem maliyeti / execution problemleri

- **Slippage ve spread modeli tamamen yok** — `trading/costs.py` sadece
  komisyon + BSMV hesaplıyor (`calculate_fee`), `check_position_size` da
  sadece asgari pozisyon büyüklüğü kontrolü.
- **Execution fiyatı gerçekçi değil:** `trading/engine.py::_latest_close()`
  hem tahmin üretilen günün (D) kapanış fiyatını hem de o GÜNÜN alım/satım
  fiyatı olarak kullanıyor — yani "D'nin kapanışını gördükten sonra, yine
  D'nin kapanışından işlem yaptım" varsayımı. Gerçekte D'nin kapanışı
  belirlendikten sonra sadece D+1'in açılışında (ya da sonrasında) işlem
  yapılabilir. Bu, backtest sonuçlarını olduğundan iyi gösterecek bir
  "aynı-bar execution" hatasıdır (yaygın bir backtest tuzağı).
- Maliyet parametreleri config'de ama `commission_pct: 0.05` gibi tek bir
  sabit senaryo var — BASE/CONSERVATIVE/STRESS gibi karşılaştırmalı
  senaryo çalıştırma altyapısı yok.
- Expected-edge-vs-cost kontrolü yok: `check_position_size` yalnızca
  "pozisyon çok küçük mü" diye bakıyor, "beklenen getiri işlem maliyetini
  aşıyor mu" diye bakmıyor (kripto-trading-bot'taki benzer kontrolün
  BorsaSite'a hiç taşınmadığı doğrulandı — `trading/costs.py` docstring'i
  bunu açıkça söylüyor).

## 6. Risk yönetimi eksikleri

Mevcut: `max_open_positions`, `max_position_pct`, `max_portfolio_exposure_pct`,
`max_hold_days`, `min_position_value_try` — hepsi `config/trading.yaml`'da,
gerçek ve test edilmiş (`test_engine.py`, `test_portfolio.py`).

**2026-09-26 güncellemesi — stop-loss/take-profit/cooldown eklendi (commit
`1302a52`):** `config/trading.yaml::stop_loss_pct` (%7) ve `take_profit_pct`
(%15), `run_once()`'ta `sell_threshold`/`max_hold_days`'ten ÖNCE kontrol
ediliyor (fiyat bazlı çıkış artık var). `cooldown_days_after_exit` (3 gün):
satılan bir sembol cooldown dolmadan yeniden alım adayı olamıyor
(`_last_exit_date`, `trades.closed_at`'e bakıyor). Üçü de varsayılan
kapalı (0.0/0), geriye uyumlu. Testler: `test_engine.py`'de 5 yeni senaryo
(tetikleme, kapalıyken tetiklenmeme, cooldown red/geçiş), mutation-test ile
doğrulandı.

**2026-09-26 karar (Umut):** **maximum single-position loss** ayrı bir
mekanizma DEĞİL — mevcut stop-loss (%7, pozisyon bazlı fiyat çıkışı) aynı
kavramın karşılığı sayılıyor, bu madde kapandı, ek kod gerekmiyor.

**2026-09-27 çözüldü — minimum expected edge after costs (commit `f4b7efb` +
`6d54172`):** önce `pipeline/train_model.py`'ye `target_return` hedefiyle bir
XGBRegressor (`magnitude_model`) eklendi, `predictions.expected_return`'a
yazılıyor; sonra `trading/costs.py::check_expected_edge` bunu round-trip
işlem maliyeti + `min_expected_edge_pct` güvenlik payıyla karşılaştırıyor
(`portfolio.py::buy()`, `engine.py`'nin aday sorgusundan `COALESCE(...,
0.0)` ile besleniyor). Varsayılan kapalı (0.0), geriye uyumlu. 23 yeni test.

**2026-09-27 kapandı (commit `7333b48`):** `backtest/portfolio.py`/
`backtest/engine.py` artık aynı `check_expected_edge` deseniyle canlı
motoru birebir yansıtıyor. Boşluk kalmadı. 146 test yeşil.

## 7. Modelleme problemleri

- **2026-09-27 kapandı (commit `5f8532c`):** deney takibi — `pipeline/
  db.py::model_experiments` tablosu, her `train_model.run()` çağrısı
  (model_version, satır sayıları, accuracy/roc_auc/mae/r2, xgb params)
  kalıcı bir satır bırakıyor; `metrics.json` hâlâ yazılıyor (üzerine
  yazılsa da artık geçmiş kaybolmuyor). 150 test yeşil.
- **2026-09-27 kapandı (aynı commit):** feature importance —
  `train_model.py`'de gerçekten eğitilmiş sınıflandırıcının
  `feature_importances_`'ından `metrics["feature_importance"]`
  (uydurma değil, gerçek eğitim çıktısı). SHAP raporu YAPILMADI (ayrı,
  daha ağır bir bağımlılık/hesap gerektirir — istenirse ayrı bir iş).
- **Hâlâ açık:** yeni feature grubu (momentum, volatilite, trend, market
  context) hiç denenmemiş — 12 feature README'nin ilk günden beri aynı.
  Bu, tek bir kod değişikliği DEĞİL: gerçek geçmiş veriyle backtest/
  walk-forward karşılaştırması gerektiren bir deney döngüsü (yeni feature
  ekle → yeniden eğit → walk-forward'da eskiyle karşılaştır → sadece
  gerçekten ölçülebilir bir iyileşme varsa production'a al). Bu container
  boş/test verisiyle çalışıyor, gerçek BIST geçmişi yfinance'ten çekilmesi
  gerekir (`scripts/run_fetch_prices.py`) — sahte bir "iyileşti" iddiası
  üretmemek için bu turda YAPILMADI, gerçek veriyle ayrı bir tur gerekir.
- ~~Calibration hiç ölçülmemiş~~ — **bu bulgu YANLIŞ/eski çıktı (2026-09-27
  doğrulandı):** `backtest/calibration.py::ProbabilityCalibrator` zaten var
  (isotonic/sigmoid, sadece validation'da fit, OOS'ta Brier before/after
  ölçülüyor — bkz. `backtest/walk_forward.py`, `test_calibration.py`,
  `test_walk_forward.py`). Muhtemelen bu madde Faz 5 tamamlanmadan
  yazılmış ve audit güncellenmemiş. Açık soru KALDI ama farklı: bu ölçüm
  SADECE `run_walk_forward`/`run_backtest` script'i elle/manuel çalıştırınca
  üretiliyor — GH Actions'daki günlük scheduler'da (Faz 10, `.github/
  workflows/`) rutin bir parçası DEĞİL, yani canlı modelin güncel
  kalibrasyonu sürekli izlenmiyor, sadece ad-hoc sorgulanabiliyor.

## 8. Veri eksikleri / veri kalitesi

- Fiyat verisi için hiçbir sağlamlık kontrolü yok: duplicate date, missing
  date, imkânsız OHLC (`low > high` gibi), negatif volume, sıfır fiyat,
  şüpheli fiyat sıçraması — `pipeline/fetch_prices.py`'de böyle bir
  doğrulama katmanı bulunamadı.
- Sentiment tarafında: gelecek zaman damgalı haber, mükerrer haber, sembole
  bağlanmamış haber kontrolü de yok (entity_linker'daki 4 düzeltme
  YANLIŞ eşleştirmeyi çözdü, ama "veri bozuk mu" kontrolü ayrı bir konu,
  hâlâ yok).
- **2026-09-27 doğrulandı (gerçek ağ isteğiyle, `config/symbols.yaml`'daki
  gerçek BIST/US sembolleriyle):** `yfinance` (kurulu sürüm 1.7.0 —
  `requirements.txt`'teki `>=0.2.40` pin'i çok geride, ayrı bir bulgu,
  aşağıda) BIST için gerçek intraday geçmiş veri VERIYOR:
  - `Ticker(...).history(period=..., interval=...)` (mevcut
    `fetch_prices.py`'nin zaten kullandığı desen, `download()` DEĞİL —
    `download()` MultiIndex kolon döndürüyor, `.history()` tek seviyeli
    kalıyor, kod değişmeden uyumlu).
  - Gerçek pencereler (THYAO.IS ile ölçüldü): `1m` ~7-8 gün, `5m`/`15m`/
    `30m` 60 gün, `1h` **730 gün** (`period="max"` bunu YANSITMIYOR, sadece
    ~1 yıl döndürüyor — `period="730d"` açıkça istenmeli).
  - Veri kalitesi: OHLCV'de NaN yok, günler arası/hafta sonu boşlukları
    beklenen (16 saat / 2 gün 16 saat) dışında anormal boşluk yok.
  - **Gerçek tuzak:** her BIST sembolünde (test edilen 5 sembol, likit ve
    az likit fark etmeden) günde TAM 1 bar `Volume=0` geliyor — günün SON
    bar'ı (kapanış müzayedesi anlık görüntüsü, Yahoo'nun BIST'e özgü bir
    tuhaflığı). AAPL'de (ABD) bu YOK. Bu, gerçek illikiditeyle
    karıştırılıp özelliklere/target'a sızabilir — implementasyon bu son
    bar'ı ATMALI (aynı `pipeline/features.py`'deki "son satır" dikkatinin
    intraday karşılığı).
  - 5 sembole art arda gerçek istek atıldı, rate-limit/429 görülmedi —
    ama üretimde ~35 BIST + birkaç US sembolü × birden fazla interval
    düzenli çekilecekse throttling/backoff YİNE de eklenmeli, test
    edilmedi.
  - **Sonuç: Faz 9-10'un önündeki veri-sağlayıcı engeli KALKTI** — veri
    gerçekten var ve kalitesi kabul edilebilir. Kalan iş artık araştırma
    değil, mühendislik: `pipeline/fetch_prices.py`'nin intraday bar
    çekme + DB şeması (`prices_daily` değil ayrı bir `prices_intraday`
    tablosu gerekir, günlük tabloyla karıştırılmamalı) + bu bar'ları
    kullanan bir feature/model/backtest katmanı — bu hâlâ ayrı,
    büyük bir tur, ama "önce doğrulama" artık yapıldı.
  - **İlgisiz ama gerçek bir bulgu:** `requirements.txt::yfinance>=0.2.40`
    pin'i kurulu `1.7.0`'dan çok geride — büyük bir major sürüm farkı.
    Şu an `fetch_prices.py` (tek-sembol `.history()`) bu sürümle hâlâ
    çalışıyor (doğrulandı) ama pin güncellenmeli, aksi halde farklı bir
    ortamda hangi sürümün kurulacağı belirsiz.

## 9. Test kapsamı

**2026-09-27 güncel:** 150 test, ~22 dosya (audit yazıldığındaki "693 satır,
8 dosya" çok eski — Faz 6/7 çalışması boyunca büyüdü). `trading/`,
`backtest/`, `pipeline/entity_linker.py`, DB migration, API portfolio
endpoint'i, trading config yükleme, artık `pipeline/{dataset,train_model,
predict_model}.py` da kapsanıyor (aşağı bkz.).

**2026-09-27 kapandı:** dört dosyanın da artık testi var — `test_dataset.py`
(`target_up`/`target_return` son-satır NaN regresyonu + `build_dataset`
filtre uyumu, `add_technical_features`'ı da dolaylı kapsıyor),
`test_train_model.py`, `test_predict_model.py` (bu turda eklendi, bkz.
§6/§7). 150 test, 12+ dosya. Kapsanmayan tek şey feature/target/leakage
kodunun TAMAMI değil — yeni eklenen feature grubu denemesi (§7, hâlâ açık)
gündeme gelirse o da kendi testini isteyecek.

## 10. Scheduler / retraining

GitHub Actions hafta içi 16:00 UTC, `workflow_dispatch` ile elle de
tetiklenebiliyor. Model her çalıştırmada yeniden eğitiliyor (GH Actions'ın
disk kalıcılığı olmadığı için — bilinçli ve doğru bir karar, README'de de
açıklanmış). Veri yenileme / model eğitimi / tahmin / trading frekansları
şu an ayrıştırılmamış, hepsi tek pipeline çalıştırmasında birlikte oluyor —
bu, günlük granülerlikte bir sorun değil ama Faz 23'teki ayrıştırma isteğiyle
şu anki mimari çelişmiyor, sadece henüz yapılmamış.

## 11. Önerilen geliştirme sırası

Kullanıcının istediği PHASE 0–10 sırası bu bulgularla tutarlı, değişiklik
önerilmiyor. Tek ekleme: **Faz 1'e (backtest engine) başlamadan önce §2'deki
etiket hatası küçük, izole bir düzeltme olarak ayrıca ele alınmalı** —
backtest engine'in kendisi bu hatalı etiketi devralıp "doğrulanmış" gibi
görünen ama aslında bozuk bir walk-forward sonucu üretebilir.

1. **(yeni, küçük)** Son-gün target etiket hatasını izole bir commit'te
   düzelt + regresyon testi ekle (mevcut testleri bozmadan).
2. Faz 1: gerçek historical backtest engine (`backtest/`).
3. Faz 2: performans metrikleri + equity curve + benchmark.
4. Faz 3: işlem maliyeti + slippage + spread modeli (aynı-bar execution
   hatasını da burada düzelt).
5. Faz 4: walk-forward validation.
6. Faz 5: threshold + calibration (yalnızca train/validation ile). ✓
   (kod zaten var — `backtest/calibration.py`/`thresholds.py`, audit'in §7
   "calibration hiç ölçülmemiş" bulgusu yanlış çıktı, 2026-09-27'de
   düzeltildi — bkz. §7. Rutin/scheduled DEĞİL, hâlâ sadece ad-hoc.)
7. Faz 6: risk yönetimi (stop-loss/take-profit opsiyonel, ölçülerek). ✓ (2026-09-26/27, bkz. §6 güncellemesi — ikisi de kapandı: max single-position loss = stop-loss, min-expected-edge Faz 7'nin bir parçası olarak çözüldü)
8. Faz 7: feature deneyleri + model iyileştirme. Beklenen getiri büyüklüğü
   tahmini (magnitude_model) kısmı ✓ (2026-09-27, bkz. §6); calibration
   zaten vardı (yukarı bkz.); deney takibi + feature importance ✓
   (2026-09-27, bkz. §7); yeni feature grubu denemesi (momentum/volatilite/
   market context) hâlâ açık — gerçek geçmiş veriyle ölçülmesi gereken bir
   deney döngüsü, tek bir kod değişikliği değil.
9. Faz 8: dashboard/raporlama. ✓ (2026-09-27 doğrulandı — bu madde de
   audit'te işaretsizdi ama kod zaten vardı: `api/main.py` + `web/
   index.html`, geniş bir `/api/*` yüzeyi (stats/symbols/predictions/
   portfolio/trades/prices/news/chart), `backtest/reports.py` JSON/CSV/
   HTML rapor üretiyor, `test_api_portfolio.py` ile test edilmiş.)
10. Faz 9-10: intraday mimari + backtest. **Kısmen var, kısmen açık:**
    `pipeline/intraday_watch.py` gün içi çalışıyor ama GÜNLÜK bar
    kullanıyor (haber/KAP tetikli erken çıkış, kendi docstring'i bunu
    bilinçli sınır olarak yazıyor) — audit'in istediği gerçek intraday
    (5m/15m/1h) FİYAT mimarisi + onun backtest'i DEĞİL. **Veri sağlayıcı
    doğrulaması 2026-09-27'de gerçek ağ isteğiyle yapıldı ve GEÇTİ**
    (bkz. §8) — `yfinance` BIST için gerçek, kaliteli intraday veri
    veriyor (1h için 730 gün geriye). Önündeki engel kalktı; kalan iş artık
    mühendislik (DB şeması, fetch/feature/backtest katmanı) — hâlâ ayrı,
    büyük bir tur ama artık bir araştırma sorusu değil.

## 12. Bu audit'in kapsamadığı / doğrulanmadığı noktalar

- `api/main.py`'nin tüm endpoint'leri tek tek incelenmedi (sadece
  `test_api_portfolio.py` üzerinden dolaylı görüldü).
- `config/sentiment.yaml`, `config/news_feeds.yaml`, `config/symbols.yaml`
  içerikleri okunmadı (düşük risk, F0/F1 kapsamında zaten stabil).
- ~~Intraday veri sağlayıcısı test edilmedi~~ — **2026-09-27'de yapıldı**,
  bkz. §8 ve §11 madde 10.
- Gerçek DB'deki (Supabase, canlı) veri hacmi/kalitesi bu audit'te
  sorgulanmadı — bulgular kod okumasına dayanıyor, canlı veriye karşı
  ayrıca doğrulanmalı.
