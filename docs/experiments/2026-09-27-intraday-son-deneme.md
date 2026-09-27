# Intraday — son deneme (ön kayıt)

Tarih: 2026-09-27. Bu dosya deney ÇALIŞTIRILMADAN ÖNCE commit'lendi; kriterler
sonuçtan sonra değiştirilemez. Başarısız olursa intraday hattı kapatılır.

## Neden
Walk-forward (audit §11 madde 11): test IC her fold'da pozitif (0.03-0.085)
ama en uç %0.2 tahmin getiri üretmedi, model rastgeleden ayırt edilemedi.
İki hipotez: (1) IC'nin bir kısmı piyasa-geneli (zamansal) bileşen, seçimde
işe yaramaz; (2) sabit 0.998 kantili fazla uç.

## Tasarım (sabit)
- Hedef: kesitsel fazla getiri = `target_return` − aynı `ts`'deki tüm
  sembollerin ortalama `target_return`'ü. Sınıflandırıcı hedefi: fazla getiri > 0.
- Fold'lar: mevcut walk-forward ile aynı (ilk train 250 gün, val 60, test 60,
  genişleyen, son pencere kısmi).
- Giriş kantili fold İÇİNDE, yalnız val'de seçilir: ızgara {0.95, 0.98, 0.99,
  0.995, 0.998}; train'de eğitilmiş model val günlerinde simüle edilir,
  maliyet = senaryo A-gerçekçi; val'de en az 10 işlem yapan kantiller arasından
  net P&L'i en yüksek olan seçilir (hiçbiri 10 işleme ulaşmazsa 0.95).
- Strateji kuralları değişmez (günde sembol başına tek giriş, t+1 açılış dolumu,
  gün sonu kapanış, stop/take-profit config'ten).

## Maliyet (birincil)
Senaryo A-gerçekçi: Midas/Enpara 0 komisyon (Umut'un aracı kurumu) + slippage
5 bps + spread 10 bps (BIST fiyat adımı: 1 tick medyan 6-7, p90 10 bps).

## Geçme kriterleri (ÜÇÜ BİRDEN)
1. Birincil maliyetle tüm OOS dönemi net toplam getiri > 0.
2. İşlem başı brüt getirinin gün-blok bootstrap %95 GA alt sınırı > 0.
3. Model, 30 rastgele tabanın (fold başına aynı giriş oranı) en az 27'sini
   işlem başı brüt getiride geçer.

Biri bile tutmazsa: intraday kapatılır, canlıya alınmaz.

## Sonuç (çalıştırıldıktan sonra eklendi) — KALDI

Kod: `2026-09-27-intraday-son-deneme.py` (repo kökünden `PYTHONPATH=.` ile).

| Fold | Test | Seçilen kantil (val) | IC fazla / ham |
|---|---|---|---|
| 0 | 2025-02-10..05-08 | 0.995 | 0.024 / 0.017 |
| 1 | 2025-05-09..08-06 | 0.998 | 0.050 / 0.067 |
| 2 | 2025-08-07..10-30 | 0.99 | 0.034 / 0.043 |
| 3 | 2025-10-31..2026-01-23 | 0.998 | 0.020 / 0.009 |
| 4 | 2026-01-26..04-21 | 0.998 | 0.041 / 0.051 |
| 5 | 2026-04-22..07-23 | 0.995 | 0.049 / 0.030 |
| 6 | 2026-07-24..09-25 | 0.998 | -0.024 / -0.008 |

- 343 işlem, birincil maliyetle net **%-11.18**, maks. düşüş %14.9.
- İşlem başı brüt -5.1 bps (kazanma %44.3); 2025 -3.8, 2026 -7.5 bps.
- Gün-blok bootstrap %95 GA [-35.3, +29.7] bps.
- Rastgele taban medyan -13.6 bps; model 30'un 28'ini geçti.

Kriter 1 (net > 0): KALDI. Kriter 2 (GA alt > 0): KALDI. Kriter 3 (≥27/30): geçti.

Yorum: model rastgele seçimden ~8 bps daha iyi sıralıyor (gerçek ama küçük
bir bilgi), ama rastgele girişin kendisi -13.6 bps — t+1 açılışta alıp gün
sonunda satmak bu evrende ortalama olarak zaten kaybettiriyor. Model bu
dezavantajı kapatamıyor; ~15-20 bps işlem maliyetini karşılamaktan çok uzak.
**Karar (ön kayıt gereği): intraday hattı kapatıldı, canlıya alınmaz.**
