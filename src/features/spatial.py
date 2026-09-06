"""Spatial moments, eccentricity, inertia tensor, and geometric descriptors."""

from typing import Dict, Tuple
import numpy as np


def compute_spatial_moments(
    defect_mask: np.ndarray,
    center: Tuple[int, int],
    radius: int,
) -> Dict[str, float]:
    """Computes second-order spatial moments, inertia tensor eigenvalues, and eccentricity.

    Args:
        defect_mask: 2D binary defect mask.
        center: (xc, yc) center coordinate.
        radius: Wafer radius.

    Returns:
        Dictionary of spatial geometric descriptors.
    """
    coords = np.argwhere(defect_mask > 0)  # [y, x]
    num_defects = len(coords)

    if num_defects < 3:
        return {
            "defect_count": float(num_defects),
            "centroid_dist_from_center": 0.0,
            "eccentricity": 0.0,
            "inertia_ratio": 1.0,
            "defect_solidity": 0.0,
        }

    # Defect Centroid
    yc_def, xc_def = np.mean(coords[:, 0]), np.mean(coords[:, 1])
    xc_wafer, yc_wafer = center
    centroid_dist = np.sqrt((xc_def - xc_wafer) ** 2 + (yc_def - yc_wafer) ** 2) / max(radius, 1)

    # Central Moments
    dy = coords[:, 0] - yc_def
    dx = coords[:, 1] - xc_def

    mu_xx = np.mean(dx ** 2)
    mu_yy = np.mean(dy ** 2)
    mu_xy = np.mean(dx * dy)

    cov_matrix = np.array([[mu_xx, mu_xy], [mu_xy, mu_yy]])
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-8)  # numerical guard
    lambda_2, lambda_1 = eigenvalues[0], eigenvalues[1]  # lambda_1 >= lambda_2

    eccentricity = np.sqrt(max(0.0, 1.0 - (lambda_2 / lambda_1)))
    inertia_ratio = lambda_1 / lambda_2

    return {
        "defect_count": float(num_defects),
        "centroid_dist_from_center": float(centroid_dist),
        "eccentricity": float(eccentricity),
        "inertia_ratio": float(inertia_ratio),
    }
