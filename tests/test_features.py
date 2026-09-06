"""Unit tests for radial, spatial, and geometric feature extraction."""

from src.features.extractor import SpatialFeatureExtractor
from src.segmentation.defect_segmenter import DefectSegmenter
from src.segmentation.wafer_mask import WaferMaskDetector
from src.utils.synthetic import generate_synthetic_wafer


def test_center_pattern_features():
    wafer = generate_synthetic_wafer(pattern="Center", size=(128, 128), noise_level=0.0)
    mask_detector = WaferMaskDetector()
    wafer_mask, circle_params = mask_detector.detect(wafer)

    segmenter = DefectSegmenter()
    defect_mask = segmenter.segment(wafer, wafer_mask)

    extractor = SpatialFeatureExtractor(num_radial_bins=8, num_angular_bins=12)
    feats = extractor.extract(defect_mask, wafer_mask, circle_params)

    # Center pattern assertions
    assert feats["center_defect_density"] > feats["edge_defect_density"]
    assert feats["center_to_edge_ratio"] > 1.5
    assert feats["radial_density_bin_0"] > feats["radial_density_bin_7"]


def test_edge_pattern_features():
    wafer = generate_synthetic_wafer(pattern="Edge", size=(128, 128), noise_level=0.0)
    mask_detector = WaferMaskDetector()
    wafer_mask, circle_params = mask_detector.detect(wafer)

    segmenter = DefectSegmenter()
    defect_mask = segmenter.segment(wafer, wafer_mask)

    extractor = SpatialFeatureExtractor(num_radial_bins=8, num_angular_bins=12)
    feats = extractor.extract(defect_mask, wafer_mask, circle_params)

    # Edge pattern assertions
    assert feats["edge_defect_density"] > feats["center_defect_density"]
    assert feats["center_to_edge_ratio"] < 0.5
    assert feats["radial_density_bin_7"] > feats["radial_density_bin_0"]


def test_scratch_pattern_eccentricity():
    wafer = generate_synthetic_wafer(pattern="Scratch", size=(128, 128), noise_level=0.0)
    mask_detector = WaferMaskDetector()
    wafer_mask, circle_params = mask_detector.detect(wafer)

    segmenter = DefectSegmenter()
    defect_mask = segmenter.segment(wafer, wafer_mask)

    extractor = SpatialFeatureExtractor(num_radial_bins=8, num_angular_bins=12)
    feats = extractor.extract(defect_mask, wafer_mask, circle_params)

    # Scratch eccentricity assertion
    assert feats["eccentricity"] > 0.65
