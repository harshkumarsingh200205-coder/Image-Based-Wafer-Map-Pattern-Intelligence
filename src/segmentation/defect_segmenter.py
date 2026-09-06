"""Defect die segmentation using adaptive thresholding and morphological filtering."""

import cv2
import numpy as np


class DefectSegmenter:
    """Extracts binary defect die masks from normalized wafer maps."""

    def __init__(self, min_defect_area: int = 1):
        self.min_defect_area = min_defect_area

    def segment(self, image: np.ndarray, wafer_mask: np.ndarray) -> np.ndarray:
        """Segments defect dies within the active wafer disc.

        Args:
            image: 2D wafer map array.
            wafer_mask: Binary boolean mask of the active wafer region.

        Returns:
            Binary defect mask (uint8: 0 for background/passing, 1 for defect die).
        """
        # Case 1: Categorical wafer map (Value 2 = Defect Die)
        if set(np.unique(image)).issubset({0, 1, 2}):
            defect_mask = (image == 2) & wafer_mask
            return defect_mask.astype(np.uint8)

        # Case 2: Grayscale intensity map (Otsu thresholding within wafer mask)
        gray = image.copy()
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        wafer_pixels = gray[wafer_mask]
        if len(wafer_pixels) == 0 or np.all(wafer_pixels == wafer_pixels[0]):
            return np.zeros_like(gray, dtype=np.uint8)

        # Adaptive Otsu on wafer region
        thresh_val, _ = cv2.threshold(wafer_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        defect_mask = (gray >= thresh_val) & wafer_mask

        # Morphological opening to eliminate isolated 1-pixel sensor noise if needed
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        filtered = cv2.morphologyEx(defect_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        return filtered.astype(np.uint8)
