import numpy as np

from backtest.calibration import ProbabilityCalibrator


def test_sigmoid_calibration_changes_and_stays_in_range():
    p = np.array([0.05, 0.15, 0.25, 0.70, 0.80, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1])
    calibrator = ProbabilityCalibrator("sigmoid").fit(p, y)
    out = calibrator.transform(p)
    assert np.all((out >= 0) & (out <= 1))
    assert len(out) == len(p)


def test_none_is_identity():
    p = np.array([0.1, 0.5, 0.9])
    calibrator = ProbabilityCalibrator("none").fit(p, [0, 1, 1])
    assert np.allclose(calibrator.transform(p), p)


def test_rejects_out_of_range_probabilities():
    calibrator = ProbabilityCalibrator("sigmoid")
    try:
        calibrator.fit(np.array([0.1, 1.5]), np.array([0, 1]))
        assert False, "beklenen ValueError fırlatılmadı"
    except ValueError:
        pass


def test_transform_before_fit_raises():
    calibrator = ProbabilityCalibrator("sigmoid")
    try:
        calibrator.transform(np.array([0.5]))
        assert False, "beklenen RuntimeError fırlatılmadı"
    except RuntimeError:
        pass
