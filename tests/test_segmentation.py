"""Unit tests for wafer disk and defect segmentation."""

import numpy as np
from src.segmentation.wafer_mask import WaferMaskDetector
from src.segmentation.defect_segmenter import DefectSegmenter
from src.utils.synthetic import generate_synthetic_wafer


def test_wafer_mask_detector():
    wafer = generate_synthetic_wafer(pattern="Edge", size=(128, 128))
    detector = WaferMaskDetector()
    mask, (xc, yc, radius) = detector.detect(wafer)

    assert mask.shape == (128, 128)
    assert 50 <= xc <= 78
    assert 50 <= yc <= 78
    assert 40 <= radius <= 64
    assert np.sum(mask) > 5000  # Wafer area check


def test_defect_segmenter():
    wafer = generate_synthetic_wafer(pattern="Center", size=(128, 128), noise_level=0.0)
    detector = WaferMaskDetector()
    mask, _ = detector.detect(wafer)

    segmenter = DefectSegmenter()
    defect_mask = segmenter.segment(wafer, mask)

    assert defect_mask.shape == (128, 128)
    assert np.sum(defect_mask) > 50  # Center defects must be present
