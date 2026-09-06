"""Unified spatial feature extraction pipeline."""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd

from .radial import compute_radial_features
from .spatial import compute_spatial_moments
from .density import compute_cluster_density


class SpatialFeatureExtractor:
    """Extracts a comprehensive 2D spatial descriptor vector from wafer & defect masks."""

    def __init__(self, num_radial_bins: int = 8, num_angular_bins: int = 12):
        self.num_radial_bins = num_radial_bins
        self.num_angular_bins = num_angular_bins

    def extract(
        self,
        defect_mask: np.ndarray,
        wafer_mask: np.ndarray,
        circle_params: Tuple[int, int, int],
    ) -> Dict[str, float]:
        """Extracts all radial, angular, spatial moment, and cluster density features."""
        xc, yc, radius = circle_params

        # Radial & Angular profile features
        radial_feats = compute_radial_features(
            defect_mask=defect_mask,
            wafer_mask=wafer_mask,
            center=(xc, yc),
            radius=radius,
            num_radial_bins=self.num_radial_bins,
            num_angular_bins=self.num_angular_bins,
        )

        # Spatial moments & geometry
        spatial_feats = compute_spatial_moments(
            defect_mask=defect_mask,
            center=(xc, yc),
            radius=radius,
        )

        # Cluster and density features
        density_feats = compute_cluster_density(
            defect_mask=defect_mask,
            wafer_mask=wafer_mask,
        )

        # Merge dictionary
        all_features = {**radial_feats, **spatial_feats, **density_feats}
        return all_features

    def to_dataframe(self, feature_dicts: list) -> pd.DataFrame:
        """Converts list of feature dictionaries into a standardized pandas DataFrame."""
        return pd.DataFrame(feature_dicts)
