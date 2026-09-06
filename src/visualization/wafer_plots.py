"""Multi-stage visual pipeline renderer."""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_wafer_stages(
    raw_image: np.ndarray,
    preprocessed: np.ndarray,
    wafer_mask: np.ndarray,
    defect_mask: np.ndarray,
    gradcam_heatmap: Optional[np.ndarray] = None,
    title: str = "Wafer Map Pipeline Stages",
) -> plt.Figure:
    """Renders a 4 or 5-panel figure displaying the sequential inspection stages."""
    num_panels = 5 if gradcam_heatmap is not None else 4
    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4))

    # Panel 1: Raw
    axes[0].imshow(raw_image, cmap="viridis" if raw_image.ndim == 2 else None)
    axes[0].set_title("1. Raw Input")
    axes[0].axis("off")

    # Panel 2: Preprocessed
    axes[1].imshow(preprocessed, cmap="gray")
    axes[1].set_title("2. Preprocessed")
    axes[1].axis("off")

    # Panel 3: Wafer Disc Mask
    axes[2].imshow(wafer_mask, cmap="Blues")
    axes[2].set_title("3. Wafer Disc Mask")
    axes[2].axis("off")

    # Panel 4: Defect Segmentation
    axes[3].imshow(defect_mask, cmap="hot")
    axes[3].set_title("4. Defect Dies")
    axes[3].axis("off")

    # Panel 5: Grad-CAM (if available)
    if gradcam_heatmap is not None:
        axes[4].imshow(preprocessed, cmap="gray")
        axes[4].imshow(gradcam_heatmap, cmap="jet", alpha=0.55)
        axes[4].set_title("5. Grad-CAM Activation")
        axes[4].axis("off")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig
