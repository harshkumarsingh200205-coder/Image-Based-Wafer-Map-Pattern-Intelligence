"""Unit tests for wafer preprocessor and ROI extraction."""

import numpy as np
import cv2
from src.preprocessing.preprocessor import WaferPreprocessor
from src.preprocessing.validator import ImageValidator
from src.utils.synthetic import generate_synthetic_wafer


def test_preprocessor_auto_crop_rectangular_canvas():
    # Create a synthetic circular wafer (100x100)
    wafer = generate_synthetic_wafer(pattern="Center", size=(100, 100))

    # Place the circular wafer onto a 16:9 canvas (100x179, aspect ratio = 1.79)
    wide_canvas = np.zeros((100, 179), dtype=np.uint8)
    # Center the wafer horizontally
    start_x = (179 - 100) // 2
    wide_canvas[:, start_x : start_x + 100] = wafer

    # Verify that without auto_crop, validator flags the aspect ratio
    validator_strict = ImageValidator(auto_crop=False)
    is_valid, err = validator_strict.validate(wide_canvas)
    assert is_valid is False
    assert "aspect ratio is distorted" in err

    # Verify that with auto_crop enabled in validator, it passes
    validator_crop = ImageValidator(auto_crop=True)
    is_valid, err = validator_crop.validate(wide_canvas)
    assert is_valid is True
    assert err is None

    # Verify that preprocessor crops and standardizes to target_size (128, 128)
    preprocessor = WaferPreprocessor(target_size=(128, 128), auto_crop_roi=True)
    processed = preprocessor.process(wide_canvas)
    assert processed.shape == (128, 128)
    assert np.sum(processed > 0) > 0


def test_preprocessor_already_square_preserves_shape():
    wafer = generate_synthetic_wafer(pattern="Donut", size=(128, 128))
    preprocessor = WaferPreprocessor(target_size=(128, 128), auto_crop_roi=True)
    processed = preprocessor.process(wafer)
    assert processed.shape == (128, 128)
