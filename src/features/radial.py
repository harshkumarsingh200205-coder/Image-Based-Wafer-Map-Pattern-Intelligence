"""Radial annular and angular polar distribution feature engineering."""

from typing import Dict, Tuple
import numpy as np


def compute_radial_features(
    defect_mask: np.ndarray,
    wafer_mask: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    num_radial_bins: int = 8,
    num_angular_bins: int = 12,
) -> Dict[str, float]:
    """Computes concentric annular defect densities and polar angular slices.

    Args:
        defect_mask: 2D binary defect mask (1: defect, 0: pass).
        wafer_mask: 2D binary active wafer mask.
        center: (xc, yc) center coordinate.
        radius: Wafer disc radius.
        num_radial_bins: Number of concentric rings (default: 8).
        num_angular_bins: Number of angular pie slices (default: 12).

    Returns:
        Dictionary of computed radial, angular, and center/edge ratio features.
    """
    h, w = defect_mask.shape[:2]
    xc, yc = center
    radius = max(radius, 1)

    y_grid, x_grid = np.ogrid[:h, :w]
    r_dist = np.sqrt((x_grid - xc) ** 2 + (y_grid - yc) ** 2)
    norm_r = r_dist / radius  # Normalized radius [0, 1]

    angles = np.arctan2(y_grid - yc, x_grid - xc)  # [-pi, pi]
    norm_angle = (angles + np.pi) / (2 * np.pi)  # [0, 1]

    features = {}

    # 1. Concentric Radial Annular Densities
    radial_densities = []
    for i in range(num_radial_bins):
        r_inner = i / num_radial_bins
        r_outer = (i + 1) / num_radial_bins
        ring_mask = (norm_r >= r_inner) & (norm_r < r_outer) & wafer_mask
        total_dies = np.sum(ring_mask)
        defect_dies = np.sum(defect_mask[ring_mask])
        density = float(defect_dies / max(total_dies, 1))
        features[f"radial_density_bin_{i}"] = density
        radial_densities.append(density)

    # 2. Angular Sector Densities
    angular_densities = []
    for j in range(num_angular_bins):
        a_inner = j / num_angular_bins
        a_outer = (j + 1) / num_angular_bins
        sector_mask = (norm_angle >= a_inner) & (norm_angle < a_outer) & wafer_mask
        total_dies = np.sum(sector_mask)
        defect_dies = np.sum(defect_mask[sector_mask])
        density = float(defect_dies / max(total_dies, 1))
        features[f"angular_density_bin_{j}"] = density
        angular_densities.append(density)

    # 3. Center-to-Edge Ratio
    center_mask = (norm_r <= 0.30) & wafer_mask
    edge_mask = (norm_r >= 0.75) & (norm_r <= 1.0) & wafer_mask

    center_defects = np.sum(defect_mask[center_mask])
    center_total = max(np.sum(center_mask), 1)
    center_density = center_defects / center_total

    edge_defects = np.sum(defect_mask[edge_mask])
    edge_total = max(np.sum(edge_mask), 1)
    edge_density = edge_defects / edge_total

    features["center_defect_density"] = float(center_density)
    features["edge_defect_density"] = float(edge_density)
    features["center_to_edge_ratio"] = float((center_density + 1e-6) / (edge_density + 1e-6))
    features["radial_density_std"] = float(np.std(radial_densities))
    features["angular_density_std"] = float(np.std(angular_densities))

    return features
