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
