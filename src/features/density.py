"""Cluster compactness and density statistics."""

from typing import Dict
import cv2
import numpy as np


def compute_cluster_density(
    defect_mask: np.ndarray,
    wafer_mask: np.ndarray,
) -> Dict[str, float]:
    """Computes connected-component cluster counts, compactness, and global density."""
    total_wafer_dies = max(int(np.sum(wafer_mask)), 1)
    defect_dies = int(np.sum(defect_mask))
    global_density = defect_dies / total_wafer_dies

    # Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        defect_mask.astype(np.uint8), connectivity=8
    )

    # Exclude background label 0
    if num_labels <= 1:
        return {
            "global_defect_density": float(global_density),
            "cluster_count": 0.0,
            "max_cluster_area_ratio": 0.0,
            "mean_cluster_area": 0.0,
        }

    cluster_areas = stats[1:, cv2.CC_STAT_AREA]
    cluster_count = len(cluster_areas)
    max_cluster_area = float(np.max(cluster_areas))
    mean_cluster_area = float(np.mean(cluster_areas))
    max_cluster_ratio = max_cluster_area / max(defect_dies, 1)

    return {
        "global_defect_density": float(global_density),
        "cluster_count": float(cluster_count),
        "max_cluster_area_ratio": float(max_cluster_ratio),
        "mean_cluster_area": float(mean_cluster_area),
    }
