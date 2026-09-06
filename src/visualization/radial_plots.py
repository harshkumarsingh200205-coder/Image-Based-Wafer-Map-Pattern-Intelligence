"""Polar and radial annular defect profile charts."""

from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


def plot_radial_polar_profile(features: Dict[str, float]) -> plt.Figure:
    """Generates a combined Annular Defect Density chart and Polar Angular Radar."""
    radial_keys = sorted([k for k in features if k.startswith("radial_density_bin_")])
    angular_keys = sorted([k for k in features if k.startswith("angular_density_bin_")])

    radial_vals = [features[k] for k in radial_keys]
    angular_vals = [features[k] for k in angular_keys]

    fig = plt.figure(figsize=(10, 4))

    # Subplot 1: Radial Profile (Concentric Rings)
    ax1 = fig.add_subplot(1, 2, 1)
    ring_indices = list(range(1, len(radial_vals) + 1))
    ax1.bar(ring_indices, radial_vals, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
    ax1.set_xlabel("Concentric Annular Ring (1: Center -> 8: Edge)")
    ax1.set_ylabel("Defect Density")
    ax1.set_title("Radial Defect Distribution")
    ax1.set_ylim([0, max(max(radial_vals, default=0.1) * 1.25, 0.1)])
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Polar Angular Radar Chart
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    if angular_vals:
        num_angles = len(angular_vals)
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False).tolist()
        # Close the loop
        angles_closed = angles + [angles[0]]
        vals_closed = angular_vals + [angular_vals[0]]
        ax2.plot(angles_closed, vals_closed, color="#ef4444", linewidth=2)
        ax2.fill(angles_closed, vals_closed, color="#f87171", alpha=0.35)
        ax2.set_title("Angular Directional Density (360°)", y=1.08)

    plt.tight_layout()
    return fig
