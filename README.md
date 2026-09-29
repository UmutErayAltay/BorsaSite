# Borsa AI — BIST + ABD hisseleri için veri hattı, tahmin ve sanal alım-satım

![Borsa AI dashboard](docs/screenshots/dashboard.png)

## Açıklama

BIST ve ABD hisseleri için günlük fiyat ve haber verisi toplayan, duygu analizi ve
makine öğrenmesiyle 5 günlük getiri olasılığı üreten bir veri hattıdır. Veri
yfinance'den (günlük + saatlik bar), 9 RSS haber kaynağından ve KAP'ın resmi JSON
API'sinden gelir; haberler FinBERT (EN) ve Türkçe BERT ile `-1..+1` arası duygu
skoruna çevrilir. XGBoost modeli teknik göstergeleri (RSI, MACD, SMA) duygu
skoruyla birleştirip "bu hisse BIST medyanının üstünde mi" olasılığını üretir.
Tahminler, komisyon/BSMV ve risk limitleri hesaba katılan **sanal** bir portföyde
eşik-bazlı alım-satıma dönüşür — gerçek emir gönderen kod repoda yoktur. Aynı
strateji backtest motoruyla (aynı-bar ve walk-forward) geçmişe karşı sınanır ve
2026-09-27'de yapılan ön kayıtlı dürüst testte **pasif al-tuta yenilmiştir**; bu
sonuç README'de açıkça durur, gizlenmez.

> English version: [README.en.md](README.en.md)

## Kurulum

```bash
cd "borsa projesi"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

`.env` içine `DATABASE_URL` ve (duygu analizi için) `HF_TOKEN=hf_...` yazın
(`https://huggingface.co/settings/tokens`).

**Veritabanı zorunludur** — SQLite tamamen bırakıldı. Yerelde:

```bash
docker compose up -d db
```

`DATABASE_URL=postgresql://borsa:borsa@localhost:5432/borsa`. Şema uygulama
gerektirmez: her script ve API açılışta `pipeline/db.py::SCHEMA_SQL`'i
çalıştırır.

## Fiyat verisi çekme (F0)

İlk kurulum (varsayılan: son 2 yıl):

```bash
python scripts/run_fetch_prices.py
```

Günlük güncelleme (son 7 gün):

```bash
python scripts/run_fetch_prices.py --incremental 7
```

Diğer bayraklar: `--period` (`1y`, `2y`, `max` — `--incremental` ile birlikte
kullanılmaz), `-v`.

Sembol başına ayrı transaction: tek bir sembolün hatası tüm döngüyü veya
sayaçları bozmaz.

## Haber verisi çekme (F1)

```bash
python scripts/run_fetch_news.py
```

Test (feed başına en fazla 5 haber):

```bash
python scripts/run_fetch_news.py --max-per-feed 5
```

Özet:

```bash
python scripts/inspect_news.py
```

Kaynaklar `config/news_feeds.yaml`'da: AA, Bloomberg HT, Dünya, Google News KAP,
Google News BIST, CNBC, MarketWatch, Yahoo Finance, Google News (S&P 500 / Wall
Street).

**KAP:** `python scripts/run_kap_sync.py --days 7` — resmi web API'si
(`byCriteria`) ile bildirim çeker, `stockCodes` alanından doğrudan eşleştirme
yapar. Google News KAP akışı ek kaynak olarak kalır.

**Haber–hisse eşleştirme:** `pipeline/entity_linker.py`, regex + şirket adı +
`config/symbol_aliases.yaml`, `config/entity_blocklist.yaml` ile eleme
(`AI`, `US`, `CEO`, `IPO`, `SPK`, `KAP` gibi kısa kelimeler ticker sanılmasın).

```bash
python scripts/run_relink.py        # RSS haberlerini yeniden eşleştir
```

## Duygu analizi (F2)

İlk kurulum (modeller indirilir, ~500MB–1GB):

```bash
python scripts/clear_hf_locks.py
python scripts/run_sentiment.py --limit 20
```

Tüm haberler:

```bash
python scripts/run_sentiment.py
```

Özet:

```bash
python scripts/inspect_sentiment.py
```

Bayraklar: `--limit N` (ilk kurulum testi), `--force` (yeniden işle),
`--skip-aggregate` (günlük hisse özetini atla), `-v`.

| Dil | Model |
|-----|--------|
| EN | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — finans haberleri |
| TR | [savasy/bert-base-turkish-sentiment-cased](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased) — genel Türkçe duygu |

Çıktı: haber başına **-1 … +1** skor (`news_sentiment`); hisse + gün ortalaması
(`sentiment_daily`). Model ayarları `config/sentiment.yaml`.

İndirme takılırsa: tüm `python` süreçlerini kapatın, `clear_hf_locks.py`
çalıştırın, tekrar deneyin (aynı anda tek terminal).

## Tahmin modeli (F3)

Teknik göstergeler (RSI14, MACD, SMA20/50, getiri ve hacim oranları) +
sentiment → **XGBoost ikilisi**:

- **Sınıflandırıcı** (`prob_up`): 5 günlük `forward_return` (D+1 açılış → D+5
  kapanış) o günün **BIST medyanından** büyükse 1. Mutlak yön değil — piyasa
  geneli hareketten arındırılmış **kesitsel** etiket.
- **Regresör** (`expected_return`): aynı trade'in beklenen getiri büyüklüğü.
  Yön olasılığı tek başına "ne kadar kazanılır" sorusunu yanıtlamıyor.

```bash
python scripts/run_train.py
python scripts/run_predict.py
python scripts/inspect_predictions.py
```

Model: `data/models/xgb_up.pkl` | Tahminler: `predictions` tablosu.
Parametreler `config/model.yaml` içinde (`training::xgb` ve `training::xgb_regressor`
ayrı bloklar; walk-forward early stopping kullanır).

**Uyarı:** Olasılıklar geçmiş veriye göre eğitilmiştir; garanti değildir.

## Dashboard (F4)

```bash
pip install -r requirements-web.txt
python scripts/run_server.py
```

Tarayıcı: **http://127.0.0.1:8000** — tek sayfa, iki sekme: **Tahminler**
(sembol tablosu, hisse detayında 1h/1d/1w/1m/1y grafik + dönemlik yükseliş
grafiği + son haberler) ve **Bot** (portföy, BIST 100 karşılaştırması, açık
pozisyonlar, kapanan işlemler).

REST API:

| Uç nokta | Açıklama |
|---|---|
| `/api/health`, `/api/stats` | Sağlık ve tablo satır sayıları |
| `/api/symbols`, `/api/symbols/{ticker}` | Sembol listesi / detay (fiyat, haber, tahmin) |
| `/api/predictions` | En son tahminler (`market`, `sort`, `order`, `limit`) |
| `/api/chart/{ticker}` | Grafik verisi (`interval=1h\|1d\|1w\|1m\|1y`) |
| `/api/prices/{ticker}` | Günlük fiyat serisi (`days`, 7–730) |
| `/api/news` | Haber akışı (`ticker`, `limit`) |
| `/api/portfolio` | Bakiye, açık pozisyonlar, toplam varlık |
| `/api/portfolio/live` | Aynı ama yfinance'ten anlık fiyatla ("şimdi kontrol et") |
| `/api/portfolio/history` | Günlük equity snapshot'ları (`days`) |
| `/api/portfolio/benchmark` | Portföy eğrisi, BIST 100'e (`XU100.IS`) normalize edilmiş |
| `/api/trades` | Kapanan işlemler + brüt/net K/Z, komisyon, kazanma oranı |

Saatlik grafik yfinance'dan canlı çekilir (60 gün), diğer aralıklar DB'den
resample edilir.

## Sanal alım-satım (F5)

Mevcut tahminleri (`prob_up`) kullanarak SADECE BIST hisselerinde, sahte
10.000 TL ile eşik-bazlı otomatik alım-satım yapar — gerçek emir göndermez.
Her işlemde komisyon + BSMV hesaba katılır. Her karar (alım, satım, tutma,
**red**) `trade_decisions` tablosuna insan-okunur gerekçeyle yazılır.

```bash
python scripts/run_trading.py
```

Dashboard'daki "Bot" sekmesi güncel bakiyeyi, açık pozisyonları, kapanan
işlemleri (brüt/net kâr, ödenen komisyon ayrı ayrı) ve portföy eğrisini BIST 100
ile aynı grafikte gösterir.

**Uyarı:** Tamamen simülasyondur; yatırım tavsiyesi değildir.

## Risk yönetimi (Faz 6–7)

Hepsi `config/trading.yaml`'da; varsayılanları geriye uyumlu (0.0/0 = kapalı).

| Ayar | Varsayılan | Etki |
|------|------------|------|
| `max_open_positions` | 8 | Aynı anda en fazla 8 açık pozisyon |
| `max_position_pct` | %15 | Pozisyon başı bakiye sınırı |
| `max_portfolio_exposure_pct` | %90 | Toplam nakit + pozisyon sınırı |
| `max_hold_days` | 10 | Maksimum tutma süresi |
| `stop_loss_pct` | %7 | Fiyat bazlı stop-loss |
| `take_profit_pct` | %15 | Fiyat bazlı kar-al |
| `cooldown_days_after_exit` | 3 gün | Satılan sembole yeniden giriş yasağı |
| `min_expected_edge_pct` | 0.0 | Beklenen getiri ≥ gidiş-dönüş maliyeti + güvenlik payı |
| `min_position_value_try` | 500 TL | Minimum işlem tutarı |

Çıkış kuralları sırayla değerlendirilir: **stop-loss → take-profit →
max_hold_days → `prob_up` düştü**.

Komisyon tarafı: `commission_pct: 0.0`, `bsmv_pct_of_commission: 5.0` — Midas/
Enpara için BIST komisyonsuz kurgu (banka aracı kurumuna geçilirse ~%0.2 +
BSMV + asgari ücret).

## Historical backtest (Faz 1–3)

Canlı motorun aksine, geçmiş bir tarih aralığını gerçekten yeniden oynatan
ayrı bir motor: `backtest/engine.py`. Komisyon + BSMV + slippage + spread'i
hesaba katıyor, Sharpe/Sortino/max drawdown/profit factor gibi metrikleri ve
buy&hold benchmark'ını üretiyor. **Bilinçli sınır:** TEK bir (mevcut, tüm geçmişle
eğitilmiş) model kullanıyor — bu yüzden sonuçlar in-sample'dır.

```bash
python scripts/run_backtest.py --scenario all
python scripts/run_backtest.py --start-date 2019-01-01 --end-date 2026-09-01
python scripts/run_backtest.py --no-db        # DB'ye yazma
```

Maliyet senaryoları `config/backtest.yaml::scenarios`: `base` (0/0 bps),
`conservative` (10/20), `stress` (30/50). Komisyon/BSMV canlı motorla aynı yerden
(`config/trading.yaml`) gelir. Sonuçlar `reports/` altına JSON/CSV/HTML olarak
yazılır (`backtest/reports.py`).

Detay, mimari ve diğer bilinçli sınırlar (aynı-bar execution) için:
`docs/BACKTEST.md`. Sistem denetimi: `docs/BACKTEST_AUDIT.md`.

## Walk-forward backtest (Faz 4)

Faz 1–3'ün in-sample sınırını kaldırır: gelecek verisini asla görmemiş bir
modelin geçmişte nasıl performans gösterirdiğini ölçer. Veri TRAIN →
VALIDATION → OOS pencerelerine bölünür, model her pencerede SADECE o pencerenin
TRAIN kısmıyla eğitilir, kalibrasyon ve alım eşiği SADECE VALIDATION'da seçilir
— OOS etiketleri hiçbir parametre seçiminde kullanılmaz. Sinyal D kapanışında
üretilir, işlem D+1 açılışında simüle edilir (bkz.
`backtest/walk_forward.py` docstring'i).

```bash
python scripts/run_walk_forward.py                    # config/backtest.yaml::default_scenario
python scripts/run_walk_forward.py --scenario stress   # ek slippage/spread ile
```

Pencere uzunlukları `config/backtest.yaml::walk_forward` (train 250 / val 60 /
oos 50 gün, 50 günlük adım, genişleyen pencere).

**Uyarı:** Geçmiş performans gelecekteki sonuçların garantisi değildir;
tamamen simülasyondur, yatırım tavsiyesi değildir.

## Dürüst test: strateji gerçekten değer katıyor mu? — **KALDI**

`docs/experiments/2026-09-27-gunluk-model-durust-test.md`, kriterler **sonuçtan
önce** commit'lendi (ön kayıt) ve sonra çalıştırıldı.

OOS 2019-02-05 → 2026-08-06, 25 pencere, 918 işlem, aynı 50 BIST hissesi:

| | Toplam | Yıllık |
|---|---|---|
| Strateji | %120.2 | %11.1 (Sharpe 0.62, maks. düşüş %24.9) |
| Eşit ağırlıklı al-tut (aynı 50 hisse) | %2032.2 | %50.4 |
| XU100 al-tut (fiyat endeksi) | %1246.9 | %41.4 |
| Rastgele model (30 koşu) medyanı | %226.0 | — |

Aktif getiri yıllık ort. **%-22.8**; strateji 30 rastgele modelin yalnızca 4'ünü
geçti (yani rastgeleden kötü). **Karar: günlük strateji hisse seçerek değer katmıyor;
pasif endeks yatırımı açıkça üstün.** Bilinen yanlılıklar (hayatta kalma, nakdin
faiz getirmemesi) dokümanda yazılı.

Bu README, modelin işe yaradığını iddia etmiyor — test etti, kazanamadı, sonucu
bıraktı.

## Intraday hattı (Faz 9–10) — **KAPANDI**

Günlük modeli daha sık çalıştırmak DEĞİL, gerçekten ayrı bir saatlik model
kuruldu:

```bash
python scripts/run_fetch_intraday_prices.py --interval 1h
python -m pipeline.intraday_train_model          # script yok, modül doğrudan
python scripts/run_intraday_walk_forward.py
```

101 sembol için 547.058 gerçek 1h bar (2023-10-27 → 2026-09-25) çekildi; hedef
sadece gün içidir (gece sıçraması hedeflenmez), strateji gün sonu **zorunlu**
kapanış yapar — model gece riski hakkında hiçbir fikir üretmiyor.

Ön kayıtlı test (`docs/experiments/2026-09-27-intraday-son-deneme.md`):
343 işlem, birincil maliyetle net **%-11.18**. Model rastgele seçimden ~8 bps
daha iyi sıralıyor (gerçek ama küçük bir bilgi), ama rastgele girişin kendisi
-13.6 bps — bu evrende t+1 açılışta alıp gün sonunda satmak zaten kaybettiriyor.
**Karar (ön kayıt gereği): intraday hattı kapatıldı, canlıya alınmaz.**

Kod duruyor (`pipeline/intraday_*.py`, `backtest/intraday_*.py`) — kapatıldı,
silinmedi.

## Gün içi haber izleme

Günlük döngü (F5) sadece kapanıştan sonra çalışır — gün içinde çıkan kötü bir KAP
bildirimi bir sonraki güne kadar hiç görülmez. Bu katman, günlük tahmin/alım
döngüsüne **yeni alım eklemez**; sadece açık pozisyonları gün içinde birkaç kez
KAP'a karşı tarar, güçlü olumsuz bir bildirim (Türkçe BERT skoru < eşik,
varsayılan -0.5) gelirse pozisyonu erken satar — risk azaltma katmanı, yeni bir
alım-satım stratejisi değil.

```bash
python scripts/run_intraday_watch.py                        # varsayılan eşik -0.5
python scripts/run_intraday_watch.py --negative-threshold -0.3
```

`.github/workflows/intraday-watch.yml`, BIST açıkken hafta içi saatte bir
otomatik çalıştırır (10:30–18:30 İstanbul). Gün içi fiyat akışı YOK — erken
çıkış en son bilinen kapanış fiyatından yapılmış gibi işaretlenir (bilinçli sınır,
bkz. `pipeline/intraday_watch.py` docstring'i).

## Günlük pipeline

**Haber–hisse eşleştirme + tam otomatik akış**
(fiyat → haber → KAP → eşleştir → sentiment → eğitim → tahmin → alım-satım):

```bash
python scripts/run_daily.py
```

Bayraklar: `--days N` (fiyat penceresi, varsayılan 7), `--skip-sentiment`,
`--skip-train`, `--skip-predict`, `--skip-trading`, `--no-relink`.

`run_daily.py` her çalıştırmada modeli de yeniden eğitir (XGBoost eğitimi hızlı,
birkaç saniye) — GitHub Actions'ın her seferinde sıfırdan bir checkout olması
nedeniyle eğitilmiş model dosyası hiçbir zaman kalıcı olmuyor; aksi halde predict
adımı `FileNotFoundError` ile başarısız olur. Eğitimi atlamak istersen
`--skip-train`.

Windows zamanlayıcı (her gün 19:00):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_task_scheduler.ps1
```

Loglar: `logs/daily_*.log`. Her adımın süresi ve hatası özet JSON'unda döner.

## Sembol listesi

`config/symbols.yaml` — **50 BIST** (`.IS` soneki, TRY) + **51 ABD** hissesi
(toplam 101). Piyasa değerine göre seçilmiş likit hisseler.

## Testler

```bash
pytest
```

208 test. `tests/conftest.py` her testi **gerçek yerel Postgres**'e karşı
çalıştırır ve sonda daima rollback eder (commit yok) — testler birbirini
kirl etmez, ekstra bir test-DB bağımlılığı gerekmez. `DATABASE_URL` localhost
değilse testler başlamayı reddeder.

## Canlı dağıtım

Dashboard ve günlük pipeline iki ayrı yerde çalışır — Render Cron Job'lar ödeme
bilgisi gerektirdiği için (aylık asgari $1) günlük iş GitHub Actions'a taşındı,
tamamen ücretsiz.

1. Bir Supabase projesi oluştur, `pipeline/db.py::SCHEMA_SQL`'i uygula.
2. **Dashboard (Render):** Render'da bu repoyu Blueprint (`render.yaml`) ile
   bağla, `borsa-ai-dashboard` servisine `DATABASE_URL`'i (Supabase connection
   string) elle gir. Build `requirements-web.txt`, start
   `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
3. **Günlük pipeline (GitHub Actions):** Repo → Settings → Secrets and
   variables → Actions → `DATABASE_URL` ve `HF_TOKEN` secret'larını ekle.
   `.github/workflows/daily-trading.yml` hafta içi her gün BIST kapanışından
   sonra (19:00 İstanbul = 16:00 UTC) otomatik çalışır. Actions sekmesinden
   "Run workflow" ile elle de tetiklenebilir.
   `.github/workflows/intraday-watch.yml` aynı iki secret'ı kullanır, ek
   kurulum gerektirmez.

## Faz durumu

| Faz | İçerik | Durum |
|-----|--------|-------|
| F0 | Günlük fiyat verisi (yfinance) | ✓ |
| F1 | RSS haber + KAP pipeline, haber–hisse eşleştirme | ✓ |
| F2 | FinBERT / TR BERT duygu analizi | ✓ |
| F3 | XGBoost + teknik göstergeler | ✓ |
| F4 | FastAPI + web dashboard | ✓ |
| F5 | Sanal alım-satım motoru | ✓ |
| Faz 1–3 | Historical backtest (aynı-bar, tek model) | ✓ |
| Faz 4 | Walk-forward backtest (TRAIN/VALIDATION/OOS) | ✓ |
| Faz 5 | Calibration + threshold seçimi (yalnız val) | ✓ |
| Faz 6 | Risk yönetimi (stop-loss / take-profit / cooldown) | ✓ |
| Faz 7 | Beklenen getiri büyüklüğü (magnitude model) + edge filtresi | ✓ |
| Faz 8 | Dashboard + raporlama | ✓ |
| Faz 9–10 | Intraday mimari + backtest | ✗ KAPANDI — test kredisine takıldı |

Kalan bilinçli açıklar: Faz 7'nin yeni feature grubu denemeleri (momentum /
volatilite / piyasa geneli bağlam) ve `BACKTEST_AUDIT.md §11`'deki diğer maddeler.

## Sorumluluk reddi

**Tahminler bilgilendirme amaçlıdır; yatırım tavsiyesi değildir.** Tamamen
simülasyondur, gerçek emir gönderme yoktur. Geçmiş performans gelecekteki
sonuçların garantisi değildir — bu projenin kendi testi, stratejinin pasif
al-tuta yenildiğini gösterdi.
