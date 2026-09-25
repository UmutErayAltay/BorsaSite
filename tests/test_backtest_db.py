from pipeline.db import insert_backtest_equity, insert_backtest_run, insert_backtest_trades


def test_backtest_tables_exist(conn):
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    names = {t["table_name"] for t in tables}
    assert {"backtest_runs", "backtest_trades", "backtest_equity"}.issubset(names)


def test_insert_backtest_run_and_children(conn):
    run_id = insert_backtest_run(
        conn, scenario="base", model_version="1.0", config_hash="abc123",
        start_date="2026-01-01", end_date="2026-06-01", starting_balance=10000.0,
        slippage_bps=0.0, spread_bps=0.0, metrics={"total_return_pct": 5.2},
    )
    assert run_id > 0

    insert_backtest_trades(conn, run_id, [{
        "symbol": "THYAO.IS", "entry_price": 100.0, "exit_price": 110.0, "quantity": 10.0,
        "gross_pnl": 100.0, "fees_paid": 5.0, "net_pnl": 95.0, "exit_reason": "prob_düştü",
        "opened_at": "2026-01-01", "closed_at": "2026-01-05",
    }])
    insert_backtest_equity(conn, run_id, [("2026-01-01", 10000.0), ("2026-01-02", 10050.0)])

    trades = conn.execute("SELECT * FROM backtest_trades WHERE run_id = ?", (run_id,)).fetchall()
    equity = conn.execute("SELECT * FROM backtest_equity WHERE run_id = ? ORDER BY snapshot_date", (run_id,)).fetchall()

    assert len(trades) == 1
    assert trades[0]["symbol"] == "THYAO.IS"
    assert len(equity) == 2
    assert float(equity[1]["total_value"]) == 10050.0


def test_insert_backtest_equity_upserts_same_date(conn):
    run_id = insert_backtest_run(
        conn, scenario="base", model_version=None, config_hash="x",
        start_date=None, end_date=None, starting_balance=1000.0,
        slippage_bps=0.0, spread_bps=0.0, metrics={},
    )
    insert_backtest_equity(conn, run_id, [("2026-01-01", 1000.0)])
    insert_backtest_equity(conn, run_id, [("2026-01-01", 1234.0)])  # same date, new value

    rows = conn.execute("SELECT total_value FROM backtest_equity WHERE run_id = ?", (run_id,)).fetchall()
    assert len(rows) == 1
    assert float(rows[0]["total_value"]) == 1234.0
