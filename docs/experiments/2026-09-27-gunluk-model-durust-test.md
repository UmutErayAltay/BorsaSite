# Günlük model — dürüst test (ön kayıt)

Tarih: 2026-09-27. Deney ÇALIŞTIRILMADAN ÖNCE commit'lendi; kriterler sonuçtan
sonra değiştirilemez. Soru: canlıda çalışan günlük strateji, aynı hisseleri
alıp tutmaktan daha iyi mi?

## Tasarım (sabit, mevcut kod)
- Motor: `backtest/walk_forward.py::run_walk_forward` (config/backtest.yaml:
  train 504, val 63, OOS 63, adım 63 gün; kalibrasyon + alım eşiği SADECE
  val'de; sinyal D kapanışı, işlem D+1 açılışı). Strateji kuralları
  `config/trading.yaml` (değiştirilmez).
- Veri: yfinance 10 yıllık günlük, 50 BIST hissesi (2016-09 → 2026-09).
  Geçmiş haber verisi yok → sentiment feature'ları 0; yalnız fiyat modeli
  test edilir.
- Maliyet (birincil): Midas 0 komisyon + slippage 5 bps + spread 10 bps.

## Karşılaştırmalar (aynı OOS tarihleri)
- B1 (birincil): aynı 50 hissenin eşit ağırlıklı al-tut'u (aynı veri, ilk
  OOS günü açılışında al, sonuna kadar tut, aynı giriş maliyeti).
- B2 (bilgi): XU100 al-tut (fiyat endeksi, temettüsüz — stratejinin lehine).
- Rastgele model: aynı pipeline, model olasılıkları düzgün rastgele (30 koşu).

## Geçme kriterleri (ÜÇÜ BİRDEN)
1. Strateji toplam getirisi > B1 toplam getirisi.
2. Günlük aktif getiri (strateji − B1), 20 günlük blok bootstrap, %95 GA alt
   sınırı > 0.
3. Strateji 30 rastgele modelin en az 27'sini toplam getiride geçer.

Kalırsa: canlı günlük strateji "hisse seçerek değer katıyor" iddiasını
taşıyamaz; öneri endeks/eşit ağırlıklı pasif yatırım olur.

## Bilinen yanlılıklar
- Hayatta kalma: evren bugünün 50 hissesi (strateji ve B1'i eşit etkiler).
- Nakit getirisi 0 sayılır (gerçekte para piyasası fonu faiz verir) —
  stratejinin boşta nakdi aleyhine; B1 hep tam yatırımlı.

## Sonuç (çalıştırıldıktan sonra eklendi) — KALDI (üç kriter de)

OOS 2019-02-05 → 2026-08-06, 25 pencere, 918 işlem, Midas + slip 5/spread 10.

| | Toplam | Yıllık |
|---|---|---|
| Strateji | %120.2 | %11.1 (Sharpe 0.62, maks. düşüş %24.9) |
| B1 eşit ağırlıklı al-tut (aynı 50 hisse) | %2032.2 | %50.4 |
| B2 XU100 al-tut (fiyat endeksi) | %1246.9 | %41.4 |
| Rastgele model (30 koşu) medyanı | %226.0 | — (min 81.5, max 853.0) |

Yıllara göre (strateji / B1): 2019 -0.8/31.6, 2020 -2.9/89.6, 2021 3.5/41.2,
2022 26.4/308.4, 2023 52.1/7.5, 2024 2.2/9.3, 2025 -0.9/14.6, 2026 13.3/10.1.

- Aktif getiri (strateji − B1) yıllık ort. %-22.8, 20 günlük blok bootstrap
  %95 GA [-49.9, +5.1].
- Strateji 30 rastgele modelin yalnız 4'ünü geçti (rastgeleden KÖTÜ).

Yorum: TL'de yüksek enflasyon döneminde strateji çoğu zaman nakitte kalıyor
(eşik seçici, pozisyon limiti) ve nominal %11/yıl, enflasyonun çok altında —
reel olarak ciddi kayıp. Hayatta kalma yanlılığından arınmış XU100 bile
stratejinin ~10 katı getirdi. **Karar (ön kayıt gereği): günlük strateji hisse
seçerek değer katmıyor; pasif endeks yatırımı açıkça üstün.**
