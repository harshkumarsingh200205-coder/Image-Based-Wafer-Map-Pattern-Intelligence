"""Wafer disc localization and background boundary detection."""

from typing import Tuple
import cv2
import numpy as np


class WaferMaskDetector:
    """Detects the circular silicon wafer boundary and creates an active area mask."""

    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """Isolates the wafer disc from surrounding background.

        Args:
            image: 2D normalized wafer map array.

        Returns:
            Tuple of (wafer_mask: np.ndarray (bool), (center_x, center_y, radius)).
        """
        h, w = image.shape[:2]

        # Case 1: Categorical wafer array (0: background, 1/2: wafer area)
        if set(np.unique(image)).issubset({0, 1, 2}):
            wafer_mask = image > 0
            # Compute center of mass / enclosing circle of non-zero region
            coords = np.argwhere(wafer_mask)
            if len(coords) > 0:
                yc, xc = int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1]))
                # Estimate radius from mask area: Area = pi * R^2 => R = sqrt(Area / pi)
                area = np.sum(wafer_mask)
                radius = int(np.sqrt(area / np.pi))
                return wafer_mask, (xc, yc, radius)

        # Case 2: Intensity image - threshold and find largest circular contour
        if image.dtype != np.uint8:
            gray = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            gray = image

        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (xc_f, yc_f), radius_f = cv2.minEnclosingCircle(largest_contour)
            xc, yc, radius = int(xc_f), int(yc_f), int(radius_f)
            
            # Construct binary circular mask
            y_grid, x_grid = np.ogrid[:h, :w]
            wafer_mask = (x_grid - xc) ** 2 + (y_grid - yc) ** 2 <= radius ** 2
            return wafer_mask, (xc, yc, radius)

        # Fallback: Canonical centered disc
        yc, xc = h // 2, w // 2
        radius = min(h, w) // 2 - 4
        y_grid, x_grid = np.ogrid[:h, :w]
        wafer_mask = (x_grid - xc) ** 2 + (y_grid - yc) ** 2 <= radius ** 2
        return wafer_mask, (xc, yc, radius)
