"""Backtest sonuçlarını `reports/` altına JSON/CSV/HTML olarak yazar.
Tek finansal/istatistik mantığı `backtest/metrics.py`'de duruyor — bu modül
sadece o sonuçları dosyaya seriliyor, kendi hesap yapmıyor."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from backtest.engine import BacktestResult
from backtest.metrics import summarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def _trade_to_dict(trade: Any) -> dict:
    if isinstance(trade, dict):
        return trade
    return {
        "symbol": trade.symbol,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "quantity": trade.quantity,
        "gross_pnl": trade.gross_pnl,
        "fees_paid": trade.fees_paid,
        "net_pnl": trade.net_pnl,
        "exit_reason": trade.exit_reason,
        "opened_at": trade.opened_at,
        "closed_at": trade.closed_at,
    }


def build_report(result: BacktestResult, starting_balance: float, scenario: str = "base") -> dict:
    trades = [_trade_to_dict(t) for t in result.trades]
    metrics = summarize(result.equity_curve, trades, starting_balance)
    return {
        "scenario": scenario,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "metrics": metrics,
        "trades": trades,
        "equity_curve": [{"date": d, "total_value": v} for d, v in result.equity_curve],
        "decisions": result.decisions,
    }


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def write_trades_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["symbol", "entry_price", "exit_price", "quantity", "gross_pnl", "fees_paid", "net_pnl", "exit_reason", "opened_at", "closed_at"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trade in report["trades"]:
            writer.writerow(trade)


def write_html(report: dict, path: Path) -> None:
    m = report["metrics"]
    dd = m["max_drawdown_pct"]

    def fmt(value):
        return "—" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))

    rows = "".join(
        f"<tr><td>{label}</td><td>{fmt(value)}</td></tr>"
        for label, value in [
            ("Senaryo", report["scenario"]),
            ("Dönem", f"{report['start_date'] or '—'} → {report['end_date'] or '—'}"),
            ("Başlangıç bakiyesi", m["starting_balance"]),
            ("Bitiş bakiyesi", m["ending_balance"]),
            ("Toplam getiri (%)", m["total_return_pct"]),
            ("Yıllıklandırılmış getiri (%)", m["annualized_return_pct"]),
            ("Maks. drawdown (%)", dd["max_drawdown_pct"]),
            ("Sharpe", m["sharpe_ratio"]),
            ("Sortino", m["sortino_ratio"]),
            ("Profit factor", m["profit_factor"]),
            ("Kazanma oranı (%)", m["win_rate_pct"]),
            ("Expectancy", m["expectancy"]),
            ("Ort. tutma süresi (gün)", m["average_holding_days"]),
            ("Turnover (%)", m["turnover_pct"]),
            ("Toplam ücret", m["total_fees"]),
            ("İşlem sayısı", m["num_trades"]),
        ]
    )
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><title>Backtest Raporu — {report['scenario']}</title>
<style>body{{font-family:sans-serif;margin:2rem;}}table{{border-collapse:collapse;}}
td{{padding:.3rem .8rem;border-bottom:1px solid #ddd;}}td:first-child{{font-weight:600;}}</style>
</head><body>
<h1>Backtest Raporu — {report['scenario']}</h1>
<table>{rows}</table>
<p><em>Bu bir simülasyondur, yatırım tavsiyesi değildir. Faz 1 modeli tüm geçmişle
eğitilmiştir (in-sample) — gerçek out-of-sample performans walk-forward
olmadan garanti edilemez.</em></p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_all(result: BacktestResult, starting_balance: float, scenario: str = "base", out_dir: Path | None = None) -> dict:
    out_dir = out_dir or DEFAULT_REPORTS_DIR
    report = build_report(result, starting_balance, scenario)
    write_json(report, out_dir / f"backtest_{scenario}.json")
    write_trades_csv(report, out_dir / f"backtest_{scenario}.csv")
    write_html(report, out_dir / f"backtest_{scenario}.html")
    write_json(report, out_dir / "backtest_latest.json")
    write_trades_csv(report, out_dir / "backtest_latest.csv")
    write_html(report, out_dir / "backtest_latest.html")
    return report
