from unittest.mock import patch

from pipeline.daily_pipeline import run


def test_daily_pipeline_calls_trading_step_by_default():
    with (
        patch("pipeline.fetch_prices.run", return_value={}),
        patch("pipeline.fetch_news_rss.run", return_value={}),
        patch("pipeline.kap_sync.sync_kap_disclosures", return_value={}),
        patch("pipeline.entity_linker.relink_all_news", return_value={}),
        patch("pipeline.db.rebuild_sentiment_daily", return_value=0),
        patch("pipeline.analyze_sentiment.run", return_value={}),
        patch("pipeline.predict_model.run", return_value={}),
        patch("pipeline.daily_pipeline._run_trading_step", return_value={"bought": 1}) as trading_mock,
    ):
        result = run(skip_train=True)

    trading_mock.assert_called_once()
    assert result["steps"]["trading"]["ok"] is True
    assert result["steps"]["trading"]["result"] == {"bought": 1}


def test_daily_pipeline_skips_trading_when_requested():
    with (
        patch("pipeline.fetch_prices.run", return_value={}),
        patch("pipeline.fetch_news_rss.run", return_value={}),
        patch("pipeline.kap_sync.sync_kap_disclosures", return_value={}),
        patch("pipeline.entity_linker.relink_all_news", return_value={}),
        patch("pipeline.db.rebuild_sentiment_daily", return_value=0),
        patch("pipeline.analyze_sentiment.run", return_value={}),
        patch("pipeline.predict_model.run", return_value={}),
        patch("pipeline.daily_pipeline._run_trading_step") as trading_mock,
    ):
        result = run(skip_train=True, skip_trading=True)

    trading_mock.assert_not_called()
    assert "trading" not in result["steps"]
