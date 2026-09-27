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
