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

1. **Faz 1 tek bir model kullanıyor — walk-forward YOK.** `run_backtest()`
   `data/models/xgb_up.pkl`'yi (tüm geçmişle eğitilmiş, backtest penceresini de
   İÇEREN) yüklüyor. Yani şu anki sonuçlar **in-sample**'dır: model, test
   edildiği dönemi zaten "görmüş" durumda. Gerçek out-of-sample performans
   walk-forward validation (Faz 4, henüz yapılmadı) olmadan iddia edilemez —
   bkz. `docs/BACKTEST_AUDIT.md` §31, Kısıt #14.
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

## Test

```bash
python -m pytest tests/test_backtest_costs.py tests/test_backtest_portfolio.py \
    tests/test_backtest_engine.py tests/test_backtest_metrics.py \
    tests/test_backtest_benchmark.py tests/test_backtest_db.py -q
```

`test_backtest_engine.py` gerçek fiyat geçmişi seed'ler ama sabit-olasılıklı
sahte bir model kullanır (gerçek XGBoost eğitimi gerektirmez) — motorun kendi
gün-gün karar/maliyet/equity döngüsünü, model kalitesinden bağımsız test eder.
