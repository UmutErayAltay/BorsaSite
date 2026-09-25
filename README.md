# Borsa AI — F0 + F1 Veri Toplama

BIST ve ABD hisseleri için günlük fiyat (F0) ve RSS haber pipeline (F1).

## Kurulum

```bash
cd "borsa projesi"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Fiyat verisi çekme

İlk kurulum (son 2 yıl):

```bash
python scripts/run_fetch_prices.py
```

Günlük güncelleme (son 7 gün):

```bash
python scripts/run_fetch_prices.py --incremental 7
```

Veritabanı: PostgreSQL. Yerelde `docker compose up -d db`, canlıda Supabase.
`.env`'de `DATABASE_URL` (bkz. `.env.example`).

## Haber verisi çekme (F1)

```bash
python scripts/run_fetch_news.py
```

Test (feed başına 5 haber):

```bash
python scripts/run_fetch_news.py --max-per-feed 5
```

Özet:

```bash
python scripts/inspect_news.py
```

Kaynaklar: `config/news_feeds.yaml` (AA, Bloomberg HT, Dünya, Google News KAP/BIST, CNBC, MarketWatch, Yahoo Finance).

**KAP:** `python scripts/run_kap_sync.py` — resmi web API (`byCriteria`) ile bildirim + `stockCodes` doğrudan eşleştirme. Google News KAP akışı ek kaynak olarak kalır.

Hisse eşleştirme: `config/symbol_aliases.yaml` + şirket adları/ticker'lar.

## Duygu analizi (F2)

İlk kurulum (modeller indirilir, ~500MB–1GB):

```bash
pip install -r requirements.txt
copy .env.example .env
# .env içine HF_TOKEN=hf_... yazın (huggingface.co/settings/tokens)
python scripts/clear_hf_locks.py
python scripts/run_sentiment.py --limit 20
```

İndirme takılırsa: tüm `python` süreçlerini kapatın, `clear_hf_locks.py` çalıştırın, tekrar deneyin (aynı anda tek terminal).

Tüm haberler:

```bash
python scripts/run_sentiment.py
```

Özet:

```bash
python scripts/inspect_sentiment.py
```

| Dil | Model |
|-----|--------|
| EN | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) |
| TR | [savasy/bert-base-turkish-sentiment-cased](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased) |

Çıktı: haber başına **-1 … +1** skor (`news_sentiment`); hisse+gün ortalaması (`sentiment_daily`).

## Tahmin modeli (F3)

Teknik göstergeler (RSI, MACD, SMA) + sentiment → XGBoost ile **yarın yükselme olasılığı**.

```bash
pip install xgboost scikit-learn joblib
python scripts/run_train.py
python scripts/run_predict.py
python scripts/inspect_predictions.py
```

Model: `data/models/xgb_up.pkl` | Tahminler: `predictions` tablosu.

**Uyarı:** Olasılıklar geçmiş veriye göre eğitilmiştir; garanti değildir.

## Dashboard (F4)

```bash
pip install fastapi uvicorn
python scripts/run_server.py
```

Tarayıcı: **http://127.0.0.1:8000**

## Günlük pipeline ve eşleştirme (iyileştirme)

**Haber–hisse eşleştirme** (regex + şirket adı + genişletilmiş alias):

```bash
python scripts/run_kap_sync.py      # KAP bildirimleri (önerilen)
python scripts/run_relink.py        # RSS haberleri yeniden eşleştir
```

**Günlük otomatik akış** (fiyat → haber → eşleştir → sentiment → tahmin):

```bash
python scripts/run_daily.py
```

`run_daily.py` her çalıştırmada modeli de yeniden eğitir (XGBoost eğitimi
hızlı, birkaç saniye) — GitHub Actions'ın her seferinde sıfırdan bir checkout
olması nedeniyle eğitilmiş model dosyası hiçbir zaman kalıcı olmuyor.
Eğitimi atlamak istersen:

```bash
python scripts/run_daily.py --skip-train
```

Windows zamanlayıcı (her gün 19:00):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_task_scheduler.ps1
```

Loglar: `logs/daily_*.log`

- Tahmin tablosu (BIST / ABD filtresi)
- Hisse detayı: fiyat grafiği, haberler, yükselme olasılığı
- REST API: `/api/predictions`, `/api/symbols/{ticker}`, `/api/prices/{ticker}`, `/api/news`

## Sembol listesi

`config/symbols.yaml` — 50 BIST (`.IS`) + 50 ABD hissesi.

## PostgreSQL (opsiyonel)

```bash
docker compose up -d db
```

`DATABASE_URL=postgresql://borsa:borsa@localhost:5432/borsa` — PostgreSQL entegrasyonu sonraki fazda.

## Sanal alım-satım (F5)

Mevcut tahminleri (`prob_up`) kullanarak SADECE BIST hisselerinde, sahte
100.000 TL ile eşik-bazlı otomatik alım-satım yapar — gerçek emir
göndermez. Her işlemde komisyon + BSMV gerçekçi şekilde hesaba katılır.
Parametreler: `config/trading.yaml`.

```bash
python scripts/run_trading.py
```

Dashboard'daki "Portföy" bölümü güncel bakiyeyi, açık pozisyonları ve
kapanan işlemleri (brüt/net kâr, ödenen komisyon ayrı ayrı) gösterir.

**Uyarı:** Tamamen simülasyondur; yatırım tavsiyesi değildir.

## Historical backtest (Faz 1-3)

Canlı motorun aksine, geçmiş bir tarih aralığını gerçekten yeniden oynatan
ayrı bir motor: `backtest/engine.py`. Komisyon+BSMV+slippage+spread'i hesaba
katıyor, Sharpe/Sortino/max drawdown/profit factor gibi metrikleri ve
buy&hold benchmark'ını üretiyor. **Bilinçli sınır:** TEK bir (mevcut,
tüm geçmişle eğitilmiş) model kullanıyor — bu yüzden sonuçlar in-sample'dır.

```bash
python scripts/run_backtest.py --scenario all
```

Detay, mimari ve diğer bilinçli sınırlar (aynı-bar execution) için:
`docs/BACKTEST.md`. Sistem denetimi: `docs/BACKTEST_AUDIT.md`.

## Walk-forward backtest (Faz 4)

Faz 1-3'ün in-sample sınırını kaldırır: gelecek verisini asla görmemiş bir
modelin geçmişte nasıl performans gösterirdiğini ölçer. Veri
TRAIN → VALIDATION → OOS pencerelerine bölünür, model her pencerede SADECE
o pencerenin TRAIN kısmıyla eğitilir, kalibrasyon ve alım eşiği SADECE
VALIDATION'da seçilir — OOS (out-of-sample) etiketleri hiçbir parametre
seçiminde kullanılmaz. Sinyal D kapanışında üretilir, işlem D+1 açılışında
simüle edilir (bkz. `backtest/walk_forward.py` docstring'i).

```bash
python scripts/run_walk_forward.py                    # config/backtest.yaml::default_scenario
python scripts/run_walk_forward.py --scenario stress   # ek slippage/spread ile
```

Pencere uzunlukları: `config/backtest.yaml::walk_forward`. Maliyet senaryoları
(`base`/`conservative`/`stress`) Faz 1-3 ile ortak (`config/backtest.yaml::
scenarios`, `backtest/costs.py`) — komisyon/BSMV `config/trading.yaml`'dan
aynen gelir, burada sadece backtest'e özgü slippage/spread eklenir.

**Uyarı:** Geçmiş performans gelecekteki sonuçların garantisi değildir;
tamamen simülasyondur, yatırım tavsiyesi değildir.

## Canlı dağıtım

Dashboard ve günlük pipeline iki ayrı yerde çalışır — Render Cron Job'lar
ödeme bilgisi gerektirdiği için (aylık asgari $1) günlük iş GitHub Actions'a
taşındı, tamamen ücretsiz.

1. Bir Supabase projesi oluştur, `pipeline/db.py::SCHEMA_SQL`'i uygula.
2. **Dashboard (Render):** Render'da bu repoyu Blueprint (`render.yaml`) ile
   bağla, `borsa-ai-dashboard` servisine `DATABASE_URL`'i (Supabase
   connection string) elle gir.
3. **Günlük pipeline (GitHub Actions):** Repo → Settings → Secrets and
   variables → Actions → `DATABASE_URL` ve `HF_TOKEN` secret'larını ekle.
   `.github/workflows/daily-trading.yml` hafta içi her gün BIST
   kapanışından sonra (19:00 İstanbul = 16:00 UTC) otomatik çalışır: fiyat
   → haber → eşleştirme → sentiment → tahmin → alım-satım. Pazartesi
   günleri model de otomatik yeniden eğitilir. Actions sekmesinden "Run
   workflow" ile elle de tetiklenebilir.

## Sonraki fazlar

| Faz | İçerik |
|-----|--------|
| F1 | RSS haber pipeline ✓ |
| F2 | FinBERT / TR BERT duygu analizi ✓ |
| F3 | XGBoost + teknik göstergeler ✓ |
| F4 | FastAPI + web dashboard ✓ |
| F5 | Sanal alım-satım motoru ✓ |
| Faz 1-3 | Historical backtest (aynı-bar, tek model) ✓ |
| Faz 4 | Walk-forward backtest (TRAIN/VALIDATION/OOS) ✓ |

**Uyarı:** Tahminler bilgilendirme amaçlıdır; yatırım tavsiyesi değildir.
