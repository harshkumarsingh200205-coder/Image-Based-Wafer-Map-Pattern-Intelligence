"""Unit tests for image and array input validator."""

# pyrefly: ignore [missing-import]
import numpy as np
from src.preprocessing.validator import ImageValidator
from src.utils.synthetic import generate_synthetic_wafer


def test_validator_valid_synthetic_wafer():
    validator = ImageValidator(min_resolution=(32, 32))
    wafer = generate_synthetic_wafer(pattern="Center", size=(128, 128))
    is_valid, err = validator.validate(wafer)
    assert is_valid is True
    assert err is None


def test_validator_low_resolution_rejected():
    validator = ImageValidator(min_resolution=(64, 64))
    small_wafer = np.ones((20, 20), dtype=np.uint8)
    is_valid, err = validator.validate(small_wafer)
    assert is_valid is False
    assert "below the minimum required" in err


def test_validator_distorted_aspect_ratio():
    validator = ImageValidator(max_aspect_ratio_diff=0.25)
    distorted = np.ones((100, 200), dtype=np.uint8)
    is_valid, err = validator.validate(distorted)
    assert is_valid is False
    assert "aspect ratio is distorted" in err


def test_validator_blank_image():
    validator = ImageValidator()
    blank = np.zeros((100, 100), dtype=np.uint8)
    is_valid, err = validator.validate(blank)
    assert is_valid is False
    assert "zero variance" in err
