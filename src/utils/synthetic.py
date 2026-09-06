"""Synthetic Wafer Map Generator for Unit Testing and Verification.

Generates mathematically controlled wafer maps across canonical failure patterns:
- Normal (clean)
- Center
- Donut
- Edge
- Ring
- Localized_Cluster
- Scratch
- Random
- Mixed
"""

from typing import Optional, Tuple
import numpy as np


def generate_synthetic_wafer(
    pattern: Optional[str] = "Center",
    size: Tuple[int, int] = (128, 128),
    noise_level: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Generates a synthetic 2D wafer array.

    Values:
        0: Background (outside wafer disk)
        1: Passing / Good Die
        2: Defective Die
    """
    rng = np.random.RandomState(seed)
    h, w = size
    yc, xc = h // 2, w // 2
    radius = min(h, w) // 2 - 4

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    # Initialize wafer area: 0 for outside, 1 for passing dies
    wafer = np.zeros((h, w), dtype=np.uint8)
    wafer_mask = dist_from_center <= radius
    wafer[wafer_mask] = 1

    pattern_clean = (pattern or "Center").strip().lower()

    if pattern_clean == "normal":
        pass  # Just nominal random noise added below

    elif pattern_clean == "center":
        # Dense spot in the inner 25% radial region
        center_mask = (dist_from_center <= radius * 0.28) & wafer_mask
        defect_prob = 0.75 * np.exp(-((dist_from_center / (radius * 0.28)) ** 2))
        defects = (rng.rand(h, w) < defect_prob) & center_mask
        wafer[defects] = 2

    elif pattern_clean == "donut":
        # Defect ring midway, clear center and clear outer edge
        donut_mask = (dist_from_center >= radius * 0.35) & (dist_from_center <= radius * 0.65) & wafer_mask
        defects = (rng.rand(h, w) < 0.65) & donut_mask
        wafer[defects] = 2

    elif pattern_clean == "edge":
        # Defects along outer perimeter (r > 0.82 R)
        edge_mask = (dist_from_center >= radius * 0.82) & wafer_mask
        defects = (rng.rand(h, w) < 0.70) & edge_mask
        wafer[defects] = 2

    elif pattern_clean == "ring":
        # Narrow continuous annular defect band
        ring_mask = (dist_from_center >= radius * 0.50) & (dist_from_center <= radius * 0.70) & wafer_mask
        defects = (rng.rand(h, w) < 0.80) & ring_mask
        wafer[defects] = 2

    elif pattern_clean in ("localized_cluster", "cluster"):
        # Dense off-center cluster
        cluster_xc = xc + int(radius * 0.45)
        cluster_yc = yc - int(radius * 0.30)
        dist_cluster = np.sqrt((x - cluster_xc) ** 2 + (y - cluster_yc) ** 2)
        cluster_mask = (dist_cluster <= radius * 0.22) & wafer_mask
        defects = (rng.rand(h, w) < 0.75) & cluster_mask
        wafer[defects] = 2

    elif pattern_clean == "scratch":
        # Linear scratch trajectory
        line_mask = (np.abs((x - xc) - 1.8 * (y - yc)) <= 2.5) & (dist_from_center <= radius * 0.85) & wafer_mask
        defects = (rng.rand(h, w) < 0.85) & line_mask
        wafer[defects] = 2

    elif pattern_clean == "random":
        # Elevated uniform random defect rate
        defects = (rng.rand(h, w) < 0.15) & wafer_mask
        wafer[defects] = 2

    elif pattern_clean == "mixed":
        # Combination of Ring + Scratch
        ring_mask = (dist_from_center >= radius * 0.50) & (dist_from_center <= radius * 0.70) & wafer_mask
        scratch_mask = (np.abs((x - xc) - 2.0 * (y - yc)) <= 2.0) & wafer_mask
        defects = ((rng.rand(h, w) < 0.60) & ring_mask) | ((rng.rand(h, w) < 0.75) & scratch_mask)
        wafer[defects] = 2

    else:
        raise ValueError(f"Unknown pattern class: {pattern}")

    # Add background defect noise across wafer
    if noise_level > 0:
        noise = (rng.rand(h, w) < noise_level) & wafer_mask
        wafer[noise] = 2

    return wafer
