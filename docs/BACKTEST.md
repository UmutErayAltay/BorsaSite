# Backtest Motoru (Faz 1-3)

`backtest/` paketi, `trading/engine.py`'nin (canlı, sadece "en güncel satır"a
bakan) yanına, geçmiş bir tarih aralığını gerçekten yeniden oynatabilen ayrı
bir motor ekler. Detaylı sistem denetimi için bkz. `docs/BACKTEST_AUDIT.md`.

## Çalıştırma

```bash
python scripts/run_backtest.py                          # BASE senaryosu, tüm geçmiş
python scripts/run_backtest.py --scenario all            # base + conservative + stress
python scripts/run_backtest.py --start-date 2026-01-01 --end-date 2026-06-01
python scripts/run_backtest.py --no-db                   # sadece dosyaya yaz, DB'ye kaydetme
```

Model önceden eğitilmiş olmalı (`python scripts/run_train.py`). Çıktılar:
`reports/backtest_<senaryo>.{json,csv,html}` + `backtest_runs`/`backtest_trades`/
`backtest_equity` tabloları (DB'ye kaydetme `--no-db` ile atlanabilir).

## Mimari

```
pipeline.dataset.build_dataset(require_target=False)   # <= o güne kadar bilinen feature'lar
        ↓
mevcut eğitilmiş model → prob_up (her satır için, tek seferde)
        ↓
backtest.engine.run_backtest()                          # günü güne sırayla oynatır
        ↓
backtest.portfolio.BacktestPortfolio                     # trading/costs.py'yi reuse eder
        ↓
backtest.metrics.summarize() + backtest.benchmark        # equity curve → metrikler
        ↓
backtest.reports.write_all()                              # JSON/CSV/HTML
```

`backtest/costs.py`, `backtest/portfolio.py` **komisyon/BSMV/pozisyon-büyüklüğü
hesabı için `trading/costs.py`'yi aynen kullanır** — finansal mantığın iki farklı
implementasyonu yok, sadece durumu bellekte tutuyor (DB'siz) ve üzerine
slippage/spread ekliyor.

## Maliyet senaryoları (`config/backtest.yaml`)

| Senaryo | slippage_bps | spread_bps |
|---|---|---|
| base | 0 | 0 |
| conservative | 10 | 20 |
| stress | 30 | 50 |

`--scenario all` üçünü de çalıştırıp ayrı raporlar üretir — gross/net getiri
farkını görmek için `backtest_base.json` ile `backtest_stress.json`'daki
`total_fees`/`net_pnl` alanlarını karşılaştır.

## BİLİNÇLİ SINIRLAR (önemli, okumadan sonuç yorumlama)

1. **Faz 1 tek bir model kullanıyor — in-sample.** `run_backtest()`
   `data/models/xgb_up.pkl`'yi (tüm geçmişle eğitilmiş, backtest penceresini de
   İÇEREN) yüklüyor. Bu sonuçlar **in-sample**'dır: model, test edildiği dönemi
   zaten "görmüş" durumda. Gerçek out-of-sample cevap için Faz 4'e (aşağıda)
   bak — orada tersine dönüyor.
2. **"Aynı-bar execution"** — işlem, tahminin üretildiği GÜNÜN kendi kapanışında
   yürütülüyor; bu, `trading/engine.py`'nin (canlı) mevcut, bilinen davranışıyla
   birebir aynı. Bu varsayımın kendisini (ör. D+1 açılışına kaydırmak)
   değiştirmek bilinçli olarak bu fazın kapsamı dışında bırakıldı — canlı
   motorun davranışını sessizce değiştirmemek için. Bkz. audit §5.
3. **Expected-edge-vs-cost kontrolü yok** (audit §14) — bir işlem sadece
   `prob_up > buy_threshold` olduğu için açılıyor, beklenen getirinin işlem
   maliyetini aşıp aşmadığına bakılmıyor.
4. Threshold optimizasyonu (Faz 9), probability calibration (Faz 5), risk
   yönetimi genişletmeleri (Faz 6, stop-loss/take-profit) ve feature deneyleri
   (Faz 7) henüz yapılmadı.

## Gerçek sonuç — Faz 4 walk-forward, Supabase canlı veri (2026-09-25)

İlk gerçek out-of-sample ölçüm. Config'teki orijinal pencereler
(train=504/validation=63/oos=63=630 gün) gerçek veriye sığmadı — `bist_only`
filtresi + SMA50 warmup + `target_up`/`open` dropna sonrası yalnızca **466
farklı `feature_date`** var (veri 2024-11-20'den başlıyor). Pencereler
466'ya sığacak şekilde küçültüldü: `train_days=250, validation_days=110,
oos_days=100, step_days=100` (`config/backtest.yaml`) — tek pencere üretiyor,
veri arttıkça büyütülüp birden fazla pencereye çıkarılabilir.

**Model kalitesi (gerçek 46.674 satır, 101 sembol, %80/%20 temporal split):**
ROC AUC **0.5066**, accuracy **0.5001** — rastgele tahminden ayırt edilemiyor.

**Walk-forward OOS sonucu** (tek pencere, TRAIN→VALIDATION→OOS, threshold
SADECE validation'da seçildi, OOS 2026-04-24 → 2026-09-14, ~4.7 ay, 100
kapanmış işlem):

| Senaryo | Bitiş bakiyesi (10.000 TL'den) | Getiri |
|---|---|---|
| base (maliyetsiz) | 8.973 TL | **-10.27%** |
| conservative | 8.668 TL | **-13.32%** |
| stress | 8.159 TL | **-18.41%** |

**Maliyetsiz senaryoda bile para kaybediliyor** — yani sonuç bir
komisyon/slippage sorunu değil, doğrudan modelin edge'i olmaması. Faz 1-3'ün
BASE senaryosunda görülen +254% (in-sample) bu yüzden gerçek değil: aynı model
hem eğitim hem test verisini gördüğü için gürültüyü ezberlemiş görünüyor,
walk-forward (gerçekten görmediği veri) bunu tersine çeviriyor.

**Neden edge yok (olası kök nedenler, doğrulanmadı — Faz 7 kapsamı):**
- Sentiment verisi çok seyrek: 101 sembol × ~2 yıl için sadece 469
  haber-sembol eşleşmesi (`news_symbol_links`) var — `sentiment_avg_3d`/
  `sentiment_news_3d` satırların büyük çoğunluğunda muhtemelen sıfır/boş,
  gerçek sinyal taşımıyor.
- 12 feature'ın hepsi teknik gösterge + fiyat türevi (RSI/MACD/SMA/return/
  volume) — hepsi zaten piyasada fiyatlanmış, kamuya açık bilgi; sadece bu
  bilgiyle bireysel hisse yönünü günlük tahmin etmek zaten literatürde zayıf
  bilinen bir problem (EMH'ye yakın davranış).
- Piyasa/endeks bazlı hiçbir context feature yok (BIST100 günlük getirisi,
  sektör bazlı hareket) — model her sembolü izole görüyor.
- ~2 yıllık (466 gün) veri, günlük-frekans bir sinyal için istatistiksel
  olarak az; gürültü/sinyal oranı yüksek.

## Test

```bash
python -m pytest tests/test_backtest_costs.py tests/test_backtest_portfolio.py \
    tests/test_backtest_engine.py tests/test_backtest_metrics.py \
    tests/test_backtest_benchmark.py tests/test_backtest_db.py -q
```

`test_backtest_engine.py` gerçek fiyat geçmişi seed'ler ama sabit-olasılıklı
sahte bir model kullanır (gerçek XGBoost eğitimi gerektirmez) — motorun kendi
gün-gün karar/maliyet/equity döngüsünü, model kalitesinden bağımsız test eder.
