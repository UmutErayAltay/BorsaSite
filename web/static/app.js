const API = "";
let currentMarket = "";
let priceChart = null;
let changeChart = null;
let allPredictions = [];
let selectedTicker = null;
let currentInterval = "1d";
let userPickedStock = false;

async function fetchJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Haberler RSS/KAP kaynaklarından geliyor — dış, güvenilmeyen veri. innerHTML'e
// yazılmadan önce kaçışlanmalı (stored XSS'i önler), href de sadece http(s)
// şemalarına izin vermeli (javascript: URL'lerini önler).
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

function safeHref(url) {
  try {
    const parsed = new URL(url, window.location.origin);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : "#";
  } catch {
    return "#";
  }
}

function formatProb(p) {
  return `${(p * 100).toFixed(1)}%`;
}

function probColor(p) {
  if (p >= 0.55) return "var(--up)";
  if (p <= 0.45) return "var(--down)";
  return "var(--accent)";
}

function directionLabel(dir) {
  if (dir === "up") return "YUKSELIS";
  if (dir === "down") return "DUSUS";
  return "YATAY";
}

async function loadStats() {
  const s = await fetchJSON("/api/stats");
  const el = document.getElementById("stats");
  const cards = [
    ["Hisse", s.symbols],
    ["Fiyat kaydı", s.prices?.toLocaleString("tr")],
    ["Haber", s.news],
    ["Sentiment", s.sentiment],
    ["Tahmin", s.predictions],
  ];
  el.innerHTML = cards
    .map(
      ([label, value]) => `
    <div class="stat-card">
      <div class="value">${value ?? "—"}</div>
      <div class="label">${label}</div>
    </div>`
    )
    .join("");
}

function openChartPanel() {
  document.getElementById("main-layout").classList.add("stock-expanded");
  document.getElementById("chart-expanded").classList.remove("hidden");
}

function closeChartPanel() {
  document.getElementById("main-layout").classList.remove("stock-expanded");
  document.getElementById("chart-expanded").classList.add("hidden");
  document.querySelectorAll("tr.selected").forEach((r) => r.classList.remove("selected"));
  selectedTicker = null;
  userPickedStock = false;
}

function renderPredictions(items) {
  const tbody = document.getElementById("predictions-body");
  const q = document.getElementById("search").value.trim().toUpperCase();
  const filtered = q
    ? items.filter((r) => r.ticker.toUpperCase().includes(q))
    : items;

  tbody.innerHTML = filtered
    .map((row) => {
      const up = row.predicted_up === 1;
      const p = row.prob_up ?? 0;
      const sent =
        row.sentiment != null ? Number(row.sentiment).toFixed(2) : "—";
      const sel = row.ticker === selectedTicker ? "selected" : "";
      return `
      <tr data-ticker="${row.ticker}" class="${sel}">
        <td>
          <span class="ticker">${row.ticker}</span>
          <span class="market-tag">${row.market}</span>
        </td>
        <td>
          <div class="prob-bar">
            <div class="bar"><div class="fill" style="width:${p * 100}%;background:${probColor(p)}"></div></div>
            <span class="pct">${formatProb(p)}</span>
          </div>
        </td>
        <td><span class="badge ${up ? "up" : "down"}">${up ? "YUKARI" : "ASAGI"}</span></td>
        <td>${row.last_close != null ? Number(row.last_close).toFixed(2) : "—"}</td>
        <td class="${sent > 0 ? "sentiment-pos" : sent < 0 ? "sentiment-neg" : ""}">${sent}</td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => selectSymbol(tr.dataset.ticker, tr));
  });
}

async function loadPredictions() {
  const params = new URLSearchParams({ limit: 100, sort: "prob_up", order: "desc" });
  if (currentMarket) params.set("market", currentMarket);
  const data = await fetchJSON(`/api/predictions?${params}`);
  allPredictions = data.items;
  renderPredictions(allPredictions);
}

async function loadChart(ticker, interval) {
  const loading = document.getElementById("chart-loading");
  loading.classList.remove("hidden");
  try {
    const data = await fetchJSON(
      `/api/chart/${encodeURIComponent(ticker)}?interval=${interval}`
    );
    updatePeriodBadge(data);
    drawPriceChart(data);
    drawChangeChart(data);
    document.getElementById(
      "chart-caption"
    ).textContent = `${data.interval_label} kapanis fiyati`;
  } finally {
    loading.classList.add("hidden");
  }
}

function updatePeriodBadge(data) {
  const el = document.getElementById("period-change");
  const pct = data.change_pct ?? 0;
  const sign = pct > 0 ? "+" : "";
  el.textContent = `${data.interval_label}: ${sign}${pct}% (${directionLabel(data.direction)})`;
  el.className = `change-badge ${data.direction || "neutral"}`;
}

function drawPriceChart(data) {
  const ctx = document.getElementById("price-chart");
  if (priceChart) priceChart.destroy();

  const isUp = data.direction === "up";
  const isDown = data.direction === "down";
  const lineColor = isUp ? "#22c55e" : isDown ? "#ef4444" : "#3b82f6";
  const fillColor = isUp
    ? "rgba(34,197,94,0.15)"
    : isDown
      ? "rgba(239,68,68,0.15)"
      : "rgba(59,130,246,0.1)";

  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "Kapanis",
          data: data.close,
          borderColor: lineColor,
          backgroundColor: fillColor,
          fill: true,
          tension: 0.15,
          pointRadius: data.dates.length > 80 ? 0 : 2,
          pointHoverRadius: 4,
          borderWidth: 2.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => `Kapanis: ${Number(c.raw).toFixed(2)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8b9cb3", maxTicksLimit: 10, maxRotation: 0 },
          grid: { color: "rgba(42,53,72,0.6)" },
        },
        y: {
          ticks: { color: "#8b9cb3" },
          grid: { color: "rgba(42,53,72,0.6)" },
        },
      },
    },
  });
}

function drawChangeChart(data) {
  const ctx = document.getElementById("change-chart");
  if (changeChart) changeChart.destroy();

  const changes = data.period_changes || [];
  if (!changes.length) {
    changeChart = new Chart(ctx, { type: "bar", data: { labels: [], datasets: [] } });
    return;
  }

  const labels = changes.map((c) => c.date);
  const values = changes.map((c) => c.change_pct);
  const colors = values.map((v) =>
    v >= 0 ? "rgba(34,197,94,0.85)" : "rgba(239,68,68,0.85)"
  );

  changeChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Degisim %",
          data: values,
          backgroundColor: colors,
          borderRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => `${c.raw >= 0 ? "+" : ""}${Number(c.raw).toFixed(2)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8b9cb3", maxTicksLimit: 12, display: changes.length <= 40 },
          grid: { display: false },
        },
        y: {
          ticks: {
            color: "#8b9cb3",
            callback: (v) => `${v}%`,
          },
          grid: { color: "rgba(42,53,72,0.4)" },
        },
      },
    },
  });
}

async function selectSymbol(ticker, rowEl) {
  selectedTicker = ticker;
  userPickedStock = true;
  openChartPanel();

  document.querySelectorAll("tr.selected").forEach((r) => r.classList.remove("selected"));
  if (rowEl) rowEl.classList.add("selected");

  const detail = await fetchJSON(`/api/symbols/${encodeURIComponent(ticker)}`);
  const pred = detail.prediction;
  const sym = detail.symbol;

  document.getElementById("detail-title").textContent = sym.name || sym.ticker;
  document.getElementById("detail-ticker").textContent = `${sym.ticker} · ${sym.market}`;

  const probEl = document.getElementById("detail-prob");
  if (pred) {
    const p = pred.prob_up;
    probEl.textContent = `Yarin yukselme tahmini: ${formatProb(p)}`;
    probEl.style.color = probColor(p);
  } else {
    probEl.textContent = "Yarin tahmini yok";
    probEl.style.color = "var(--muted)";
  }

  await loadChart(ticker, currentInterval);

  const newsEl = document.getElementById("detail-news");
  newsEl.innerHTML = (detail.news || [])
    .map(
      (n) => `
    <li>
      <a href="${safeHref(n.url)}" target="_blank" rel="noopener">${escapeHtml(n.title)}</a>
      <div class="news-meta">${escapeHtml(n.source || "")} · ${escapeHtml(n.published_at || "")} ${
        n.score != null ? `· sentiment ${Number(n.score).toFixed(2)}` : ""
      }</div>
    </li>`
    )
    .join("") || "<li>Haber bulunamadi</li>";
}

async function loadNewsFeed() {
  const data = await fetchJSON("/api/news?limit=15");
  const el = document.getElementById("news-feed");
  el.innerHTML = data.items
    .map(
      (n) => `
    <li>
      <a href="${safeHref(n.url)}" target="_blank" rel="noopener">${escapeHtml(n.title)}</a>
      <div class="news-meta">${escapeHtml(n.source)} · ${escapeHtml(n.published_at || "")} ${
        n.sentiment_score != null
          ? `· <span class="${n.sentiment_score > 0 ? "sentiment-pos" : "sentiment-neg"}">${Number(n.sentiment_score).toFixed(2)}</span>`
          : ""
      }</div>
    </li>`
    )
    .join("");
}

async function loadPortfolio() {
  const data = await fetchJSON("/api/portfolio");
  const el = document.getElementById("portfolio-stats");
  const cards = [
    ["Nakit", `${data.balance.toLocaleString("tr")} TL`],
    ["Pozisyon değeri", `${data.positions_value.toLocaleString("tr")} TL`],
    ["Toplam varlık", `${data.total_value.toLocaleString("tr")} TL`],
    ["Başlangıç", `${data.starting_balance.toLocaleString("tr")} TL`],
  ];
  el.innerHTML = cards
    .map(
      ([label, value]) => `
    <div class="stat-card">
      <div class="value">${value}</div>
      <div class="label">${label}</div>
    </div>`
    )
    .join("");

  const tbody = document.getElementById("positions-body");
  tbody.innerHTML =
    data.open_positions
      .map((p) => {
        const pnlClass = p.unrealized_pnl >= 0 ? "sentiment-pos" : "sentiment-neg";
        return `
      <tr>
        <td>${p.ticker}</td>
        <td>${p.entry_price.toFixed(2)}</td>
        <td>${p.current_price.toFixed(2)}</td>
        <td>${p.quantity.toFixed(4)}</td>
        <td class="${pnlClass}">${p.unrealized_pnl.toFixed(2)} TL</td>
        <td>${p.opened_at}</td>
      </tr>`;
      })
      .join("") || `<tr><td colspan="6">Açık pozisyon yok.</td></tr>`;
}

async function loadTrades() {
  const data = await fetchJSON("/api/trades?limit=50");
  const tbody = document.getElementById("trades-body");
  tbody.innerHTML =
    data.items
      .map((t) => {
        const pnlClass = t.net_pnl >= 0 ? "sentiment-pos" : "sentiment-neg";
        return `
      <tr>
        <td>${t.ticker}</td>
        <td>${Number(t.entry_price).toFixed(2)}</td>
        <td>${Number(t.exit_price).toFixed(2)}</td>
        <td>${Number(t.gross_pnl).toFixed(2)}</td>
        <td>${Number(t.fees_paid).toFixed(2)}</td>
        <td class="${pnlClass}">${Number(t.net_pnl).toFixed(2)}</td>
        <td>${t.exit_reason}</td>
        <td>${t.closed_at}</td>
      </tr>`;
      })
      .join("") || `<tr><td colspan="8">Henüz kapanan işlem yok.</td></tr>`;

  const summary = document.getElementById("trades-summary");
  if (data.totals.trade_count) {
    summary.textContent =
      `Toplam ${data.totals.trade_count} işlem · ` +
      `brüt ${data.totals.total_gross_pnl.toFixed(2)} TL · ` +
      `komisyon ${data.totals.total_fees.toFixed(2)} TL · ` +
      `net ${data.totals.total_net_pnl.toFixed(2)} TL · ` +
      `kazanma oranı %${data.totals.win_rate ?? "—"}`;
  } else {
    summary.textContent = "";
  }
}

document.querySelectorAll(".filter").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentMarket = btn.dataset.market || "";
    loadPredictions();
  });
});

document.getElementById("search").addEventListener("input", () => {
  renderPredictions(allPredictions);
});

document.querySelectorAll(".interval").forEach((btn) => {
  btn.addEventListener("click", async () => {
    document.querySelectorAll(".interval").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentInterval = btn.dataset.interval;
    if (selectedTicker) await loadChart(selectedTicker, currentInterval);
  });
});

document.getElementById("btn-close-chart").addEventListener("click", closeChartPanel);

async function init() {
  try {
    await loadStats();
    await loadPredictions();
    await loadNewsFeed();
    await loadPortfolio();
    await loadTrades();
  } catch (e) {
    console.error(e);
    document.body.insertAdjacentHTML(
      "beforeend",
      `<p style="color:#ef4444;padding:2rem">API baglantisi kurulamadi. Sunucuyu baslatin: python scripts/run_server.py</p>`
    );
  }
}

init();
