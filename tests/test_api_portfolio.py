from datetime import date

from fastapi.testclient import TestClient

from api.main import app
from pipeline.db import upsert_prediction, upsert_prices, upsert_symbol
from trading.config import load_trading_config
from trading.engine import run_once
from trading.portfolio import ensure_portfolio

client = TestClient(app)


def test_portfolio_endpoint_reflects_open_position(committed_conn):
    cfg = load_trading_config()
    ensure_portfolio(committed_conn, cfg.starting_balance)
    symbol_id = upsert_symbol(committed_conn, "THYAO.IS", "BIST", "TRY", "Türk Hava Yolları")
    upsert_prices(committed_conn, symbol_id, [("2026-09-10", 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0)])
    upsert_prediction(committed_conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(committed_conn, cfg, decision_date=date(2026, 9, 10))

    response = client.get("/api/portfolio")

    assert response.status_code == 200
    data = response.json()
    assert data["open_positions"][0]["ticker"] == "THYAO.IS"
    assert data["balance"] < cfg.starting_balance


def test_trades_endpoint_reports_totals_after_a_closed_trade(committed_conn):
    cfg = load_trading_config()
    ensure_portfolio(committed_conn, cfg.starting_balance)
    symbol_id = upsert_symbol(committed_conn, "THYAO.IS", "BIST", "TRY")
    upsert_prices(committed_conn, symbol_id, [("2026-09-10", 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0)])
    upsert_prediction(committed_conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(committed_conn, cfg, decision_date=date(2026, 9, 10))

    upsert_prices(committed_conn, symbol_id, [("2026-09-11", 90.0, 90.0, 90.0, 90.0, 90.0, 1000.0)])
    upsert_prediction(committed_conn, symbol_id, "2026-09-11", "2026-09-12", 0.20, 0, "1.0")
    run_once(committed_conn, cfg, decision_date=date(2026, 9, 11))

    response = client.get("/api/trades")

    assert response.status_code == 200
    data = response.json()
    assert data["totals"]["trade_count"] == 1
    assert data["totals"]["total_fees"] > 0
