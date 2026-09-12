# Borsa AI — Sanal Alım-Satım Motoru + Canlı Dağıtım

**Tarih:** 2026-09-12
**Durum:** Onaylandı, uygulama planına geçiliyor.

## Amaç

"Borsa AI" (BIST + ABD hisseleri için haber/duygu analizi + XGBoost ile
"yarın yükselir mi" tahmini üreten mevcut sistem) şu ana kadar **salt
okunur** bir tahmin panosu. Hiçbir alım-satım kararı vermiyor, hiçbir
portföy/kâr-zarar takibi yok, ve hiç canlıya alınmamış (sadece yerel
makinede, elle/Windows Task Scheduler ile çalıştırılıyor).

Bu spesifikasyon üç şeyi kapsar:
1. Mevcut tahminleri kullanarak sahte parayla otomatik alım-satım yapan bir
   motor eklemek — her işlemde gerçekçi komisyon/vergi kesintisiyle.
2. Bu motor + mevcut pipeline'ı gerçekten canlı bir sunucuda, günlük olarak
   kendiliğinden çalışır hale getirmek.
3. Yeni trading mantığı için bir test paketi (proje şu an 0 test içeriyor).

**Kapsam dışı (bilinçli):** ABD hisseleri (sadece BIST), mevcut haber/
sentiment/ML pipeline'ına geriye dönük test yazmak (ayrı, çok daha büyük
bir iş — ağ ve model bağımlı), gerçek para/gerçek emir gönderimi (tamamen
simülasyon).

## Mimari Karar: Neden Postgres'e Geçiyoruz

Render'ın Cron Job servisleri **hiçbir şekilde kalıcı disk kullanamıyor**
(resmi dokümantasyon: "Cron jobs can't provision or access a persistent
disk") — her çalıştırmada sıfırdan bir konteynerle başlıyor. Mevcut mimari
(`data/borsa.db`, yerel SQLite) aylarca yerel makinede biriken bir dosyaya
dayanıyor; bu haliyle Render'da hiçbir şey canlı çalışamaz, sadece yeni
trading özelliği değil, mevcut fiyat/haber/tahmin pipeline'ının tamamı.

**Karar:** SQLite tamamen bırakılıyor, Supabase Postgres'e (ücretsiz katman)
geçiliyor. İki SQL lehçesini paralel desteklemek (SQLite yerel + Postgres
prod) gereksiz karmaşıklık katardı — YAGNI. Yerel geliştirme de aynı
Postgres'e bağlanacak (zaten var olan `docker-compose.yml::db` servisiyle,
ya da doğrudan Supabase'e).

Render'ın kendi ücretsiz Postgres'i yerine Supabase seçildi çünkü Render'ın
ücretsiz Postgres'i sabit bir süre (tarihsel olarak ~30 gün) sonra ücretli
plana geçmeyi zorluyor; Supabase'in ücretsiz katmanı süresiz ama haftalık
inaktivitede uyuyor — bu proje için daha iyi bir denge, ve `sosyal-medya`
projesinden zaten bilinen/tanıdık bir "uyandır" deseni var.

## Veri Katmanı Değişiklikleri

### Şema taşıma — küçültülmüş kapsam (kod incelemesiyle doğrulandı)

Kod taraması şunu gösterdi: `?` placeholder, `INSERT OR IGNORE`,
`cur.lastrowid`, `conn.total_changes` gibi SQLite'a özgü her şey **sadece
`pipeline/db.py` içinde** yaşıyor. Diğer 15 çağrı noktası (`api/main.py`,
`api/chart_data.py`, `pipeline/analyze_sentiment.py`, `pipeline/dataset.py`,
`pipeline/entity_linker.py`, `pipeline/kap_sync.py`, 5× `scripts/
inspect_*.py`) sadece düz `conn.execute(sql, params)` + `row["col"]`
(dict-tarzı) erişim kullanıyor, tuple-index erişim (`row[0]`) YOK. Bu, tüm
taşımanın **sadece `pipeline/db.py`'a dokunarak** yapılabileceği anlamına
gelir — diğer 15 dosyaya hiç dokunulmaz.

`get_connection()`, gerçek bir `psycopg` bağlantısını sarmalayan ince bir
uyumluluk sınıfı döndürür:
- `.execute(sql, params)` — `sql.replace("?", "%s")` yapıp bir cursor açar,
  çalıştırır, cursor'ı döner (sqlite3.Connection'ın `execute()` kısayolunu
  taklit eder — psycopg'de bu kısayol yok, `cursor()` gerekir)
- `row_factory=psycopg.rows.dict_row` — dönen satırlar zaten `dict`,
  `row["col"]` ve `dict(row)` mevcut kodda değişiklik gerektirmeden çalışır
- `.executemany(sql, rows)` — aynı `?`→`%s` çevirisiyle
- Varsayım: hiçbir SQL metninde placeholder OLMAYAN gerçek bir `?` karakteri
  yok (kod taramasıyla doğrulandı — hepsi ya sayı/metin karşılaştırması ya
  da LIKE deseni, literal `?` içeren yok). Test paketi (gerçek Postgres'e
  karşı) bunu kırılırsa hemen yakalar.

`pipeline/db.py::SCHEMA_SQL` (mevcut `symbols`, `prices_daily`, `news_raw`,
`news_symbol_links`, `news_sentiment`, `sentiment_daily`, `predictions`
tabloları) Postgres söz dizimine çevrilecek:
- `INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL PRIMARY KEY`
- `datetime('now')` → `NOW()`, kolon tipleri `TIMESTAMPTZ`
- `INSERT OR IGNORE INTO news_raw ...` → `INSERT INTO ... ON CONFLICT (url)
  DO NOTHING RETURNING id` (dönen satır yoksa zaten yok sayıldı demektir —
  `cur.lastrowid` yerine `RETURNING id` + `cur.fetchone()`)
- `conn.total_changes` (upsert_prices'ın dönüş değeri) → `cur.rowcount`
  toplamı (executemany sonrası psycopg'nin kendi `rowcount`'u)
- `INSERT ... ON CONFLICT(...) DO UPDATE ... excluded.col` — söz dizimi
  Postgres'te de aynı, değişiklik gerekmez

`get_connection()` bağlam yöneticisi imzası (commit/rollback/close) aynı
kalacak.

### Yeni tablolar (bu spesifikasyona özel)

```sql
CREATE TABLE portfolio (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- tek satır (singleton)
    starting_balance NUMERIC(14,2) NOT NULL,
    balance NUMERIC(14,2) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    entry_price NUMERIC(14,4) NOT NULL,
    quantity NUMERIC(14,4) NOT NULL,
    entry_prob_up REAL NOT NULL,
    opened_at DATE NOT NULL,
    entry_fee NUMERIC(14,2) NOT NULL,
    UNIQUE(symbol_id)  -- aynı sembolde tek açık pozisyon
);

CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    entry_price NUMERIC(14,4) NOT NULL,
    exit_price NUMERIC(14,4) NOT NULL,
    quantity NUMERIC(14,4) NOT NULL,
    gross_pnl NUMERIC(14,2) NOT NULL,
    fees_paid NUMERIC(14,2) NOT NULL,   -- alış+satış komisyon+BSMV toplamı
    net_pnl NUMERIC(14,2) NOT NULL,
    exit_reason TEXT NOT NULL,          -- 'prob_düştü' | 'max_hold_süresi' | 'elle'
    opened_at DATE NOT NULL,
    closed_at DATE NOT NULL
);

CREATE TABLE trade_decisions (
    id SERIAL PRIMARY KEY,
    decision_date DATE NOT NULL,
    symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    action TEXT NOT NULL,               -- 'al' | 'sat' | 'tut' | 'red'
    reason TEXT NOT NULL,               -- insan-okunur gerekçe
    prob_up REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_closed_at ON trades(closed_at DESC);
CREATE INDEX idx_decisions_date ON trade_decisions(decision_date DESC);
```

`positions`/`trades`'te fiyat ve miktarları `NUMERIC` (ondalıklı, kesin)
kullanıyoruz — `REAL`/float değil, çünkü para hesaplarında birikimli
yuvarlama hatası istemiyoruz (mevcut şemadaki `prices_daily.close REAL`
zaten piyasa verisi için var, para muhasebesi için ayrı bir hassasiyet
gerekiyor).

## Yeni Modül: `trading/`

```
trading/
  __init__.py
  config.py       # config/trading.yaml'ı yükler, tipli bir dataclass döner
  costs.py        # SAF fonksiyonlar: komisyon+BSMV hesabı, "pozisyon çok
                  # küçük mü" kontrolü — I/O yok, test etmesi en kolay katman
  portfolio.py    # Portfolio sınıfı: DB'ye karşı bakiye/pozisyon okuma-yazma
  engine.py       # TradingEngine.run_once(conn) — günlük al/sat/tut kararı
```

### `trading/config.py` — `config/trading.yaml`

```yaml
starting_balance: 100000.0        # TL, ilk kurulumda tek seferlik

buy_threshold: 0.62               # prob_up bunun üzerindeyse alım adayı
sell_threshold: 0.50              # açık pozisyonun güncel prob_up'ı bunun
                                   # altına düşerse sat
max_hold_days: 10                 # bu kadar iş günü sonra prob_up'a
                                   # bakılmaksızın kapat (durgun/kapsam dışı
                                   # kalan sembolde sonsuza dek beklememek için)

max_open_positions: 8
max_position_pct: 0.15            # yeni bir pozisyon, güncel bakiyenin en
                                   # fazla bu oranı kadar olabilir
max_portfolio_exposure_pct: 0.90  # tüm açık pozisyonların toplam maliyeti,
                                   # starting_balance'ın bu oranını aşamaz

commission_pct: 0.05              # işlem tutarının yüzdesi (alış VE satışta)
bsmv_pct_of_commission: 5.0       # komisyon üzerinden BSMV (gerçek TR
                                   # aracı kurum uygulaması)
min_commission_try: 5.0           # asgari işlem ücreti (TL)
min_position_value_try: 500.0     # bunun altındaki pozisyonlar açılmaz —
                                   # asgari ücret payı çok büyür (bkz. costs.py)
```

### `trading/costs.py` (saf, side-effect'siz — `risk/cost_check.py`'daki
disiplinin aynısı, ama bu stratejinin şekline uyarlanmış)

```python
@dataclass(frozen=True)
class CommissionResult:
    commission: float       # ham komisyon
    bsmv: float              # komisyon üzerinden BSMV
    total_fee: float         # commission + bsmv (asgari ücretle max'lanmış)

def calculate_fee(trade_value: float, cfg: TradingConfig) -> CommissionResult: ...

@dataclass(frozen=True)
class PositionSizeCheck:
    allowed: bool
    reason: str

def check_position_size(position_value: float, cfg: TradingConfig) -> PositionSizeCheck:
    """min_position_value_try altındaki pozisyonları reddeder — asgari işlem
    ücretinin pozisyon değerine oranı çok büyüyüp komisyonun kârı baştan
    yiyeceği durumu önler. kripto-trading-bot'taki 'beklenen brüt kâr >
    toplam maliyet' kontrolünün DEĞİL, bunun yerine kullanılan basitleştirilmiş
    biçimi — çünkü bu strateji stop-loss/take-profit değil, eşik-bazlı; 'ne
    kadar kâr beklendiği' yok, sadece 'bu büyüklükte işlem mantıklı mı' var."""
```

### `trading/portfolio.py`

`kripto-trading-bot/engine/portfolio.py`'daki desenle AYNI davranış
sözleşmesi (bakiye yetersizse/limit aşılırsa `(False, sebep)` döner, her
zaman insan-okunur bir `reason`), ama JSON dosyası yerine yukarıdaki
Postgres tablolarına karşı çalışır, thread-safety gerekmez (günde bir kez,
tek process'te sıralı çalışıyor — kilit mekanizması bu bağlamda gereksiz
karmaşıklık olurdu, kripto bot'un paralel-sembol işleme ihtiyacı burada yok).

- `buy(conn, symbol_id, price, prob_up, decision_date) -> (bool, str)`
- `sell(conn, symbol_id, price, exit_reason, decision_date) -> ClosedTrade | None`
- `get_state(conn) -> PortfolioState` (bakiye + açık pozisyonlar, dashboard/API için)

### `trading/engine.py`

```python
def run_once(conn, decision_date: date | None = None) -> dict:
    """Her açık pozisyonu güncel tahminine göre değerlendir (sat/tut),
    sonra buy_threshold'u geçen ve henüz pozisyonu olmayan sembolleri
    aday olarak sırala (prob_up'a göre azalan), limitler dolana kadar al.
    Her karar trade_decisions'a nedenli olarak loglanır — reddedilenler
    dahil (kripto bot'taki 'reddedilen işlemler nedensel loglanmalı'
    disiplini)."""
```

## Pipeline Entegrasyonu

`pipeline/daily_pipeline.py::run()` içine `predict` adımından sonra yeni bir
`trading` adımı eklenir (`skip_trading: bool = False` parametresiyle).
`scripts/run_daily.py`'a `--skip-trading` bayrağı eklenir (mevcut
`--skip-predict` deseniyle tutarlı). Haftalık model yenileme
(`skip_train`) artık gün kontrolüyle otomatikleşir: `run_daily.py`
Render cron'dan her gün çağrılacağı için, Pazartesi günleri `--train`
bayrağı cron komutunun kendisinde sabitlenir (crontab'da ayrı zamanlanmış
ikinci bir komut yerine, aynı script'e günü kontrol eden basit bir
`datetime.now().weekday() == 0` kontrolü — tek cron job, tek yerde mantık).

## API ve Dashboard

`api/main.py`'a iki yeni endpoint:
- `GET /api/portfolio` — bakiye, açık pozisyonlar (güncel fiyatla
  gerçekleşmemiş kâr/zarar hesaplı), toplam varlık değeri
- `GET /api/trades?limit=50` — kapanan işlemler (brüt/net kâr, ödenen
  ücret ayrı sütunlarda), toplam istatistik (toplam net kâr, toplam ödenen
  komisyon, işlem sayısı, kazanma oranı)

`web/index.html`'e mevcut desene uyan basit bir "Portföy" sekmesi: bakiye
kartı, açık pozisyon tablosu, kapanan işlem tablosu. Yeni bir grafik
kütüphanesi/altyapısı eklenmez (YAGNI) — mevcut tablo/kart bileşenleri
genişletilir.

## Test Planı

`tests/` (yeni dizin, proje kökünde), `pytest`. Yeni bir test bağımlılığı
(`pytest-postgresql` gibi ephemeral-Postgres-spinup paketleri) eklenmez —
zaten var olan `docker-compose.yml::db` yerel Postgres'ine karşı çalışılır,
her test bir transaction açıp sonunda rollback eder (commit edilmez) — gerçek
Postgres'e karşı ama testler birbirini kirletmez, ekstra bağımlılık yok.

- `test_costs.py` — `calculate_fee`/`check_position_size` saf fonksiyon
  testleri: normal işlem, asgari ücret devreye giren küçük işlem, sınırda
  değerler
- `test_portfolio.py` — `buy`/`sell`: bakiyeden doğru düşme/ekleme, yetersiz
  bakiye reddi, `max_open_positions`/`max_portfolio_exposure_pct` reddi,
  komisyonun doğru hesaba katılması (P&L testi: 100 TL'ye alıp 110 TL'ye
  satınca net kâr, brüt kârdan TAM OLARAK iki yönlü komisyon kadar az olmalı)
- `test_engine.py` — sahte `predictions` satırlarıyla: eşik üstü alım,
  eşik altı satım, `max_hold_days` zorlaması, limit dolunca red + doğru
  `trade_decisions` kaydı

Mevcut `pipeline/*.py` (haber çekme, sentiment, XGBoost eğitimi) için test
yazmak bu spesifikasyonun kapsamı DIŞINDA — ağ çağrıları ve ağır ML
modelleri gerektiriyor, ayrı ve çok daha büyük bir iş.

## Dağıtım

1. **Supabase projesi** oluşturulur, yeni şema (mevcut + yeni tablolar)
   uygulanır.
2. **Render Web Service** (ücretsiz katman) — mevcut FastAPI dashboard,
   `DATABASE_URL` Supabase'i gösterir. `render.yaml` blueprint'i bu
   projeye eklenir (sosyal-medya'daki desenle tutarlı: `sync: false` olan
   sırlar dashboard'dan elle girilir).
3. **Render Cron Job** (~$1/ay asgari ücret) — `0 16 * * 1-5` (BIST
   kapanışından sonra, hafta içi), `python scripts/run_daily.py` çalıştırır.
   Aynı `DATABASE_URL` + `HF_TOKEN` ortam değişkenleri.
4. Bilinen risk: haber/sentiment adımı her cron çalıştırmasında FinBERT/
   TR-BERT modellerini (~500MB-1GB) sıfırdan indirir (disk kalıcı değil) —
   çalışma süresi birkaç dakika sürebilir, cron dakika-bazlı ücretlendiği
   için maliyet düşük kalır (~$1-2/ay toplam) ama bu ponytail tarzı bir
   "bilinen tavan": ölçekte sorun olursa yükseltme yolu modelleri Render'ın
   build-time'ında (disk imajına) gömmek olur.

## Açık Riskler (etiketlenmiş, engelleyici değil)

- Supabase ücretsiz katmanı haftalık inaktivitede uyur — cron job zaten
  her gün bağlandığı için pratikte hiç uyumayacak, ama Render Web Service
  (dashboard) hafta sonu hiç ziyaret edilmezse uyuyabilir; `sosyal-medya`
  projesinden bilinen `restore_project` deseniyle uyandırılabilir.
- Komisyon/BSMV oranları GERÇEKÇİ AMA İLLÜSTRATİF varsayılanlar — gerçek
  bir aracı kurumun güncel tarifesini birebir yansıtmaz, `config/
  trading.yaml`'dan kolayca değiştirilebilir.
- `max_hold_days` sonrası zorunlu satış, o sembolün prob_up'ı hâlâ yüksek
  olsa bile pozisyonu kapatabilir — kasıtlı bir basitleştirme (durgun/
  kapsam dışı kalan sembollerde sonsuz bekleme riskini ortadan kaldırmak
  için), ileride "hâlâ güncel tahmini varsa yenile" mantığına
  genişletilebilir.
