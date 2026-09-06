"""Wafer map normalization and preprocessing pipeline."""

from typing import Tuple
import cv2
import numpy as np


class WaferPreprocessor:
    """Standardizes wafer map resolution, channels, and value representations with ROI auto-cropping."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (128, 128),
        auto_crop_roi: bool = True,
        roi_padding_ratio: float = 0.05,
    ):
        self.target_size = target_size
        self.auto_crop_roi = auto_crop_roi
        self.roi_padding_ratio = roi_padding_ratio

    def extract_wafer_roi(self, image: np.ndarray) -> np.ndarray:
        """Detects and tightly crops the circular wafer disc ROI from rectangular canvases or screenshots.

        Args:
            image: 2D or 3D input array (e.g. screenshot containing headers, legends, or non-square margins).

        Returns:
            Square 2D/3D cropped sub-array centered on the detected wafer disc.
        """
        h, w = image.shape[:2]
        aspect_ratio = max(h, w) / max(min(h, w), 1)

        # If already approximately square, return directly
        if aspect_ratio <= 1.05:
            return image

        # Convert to 2D grayscale for contour detection
        if image.ndim == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image[:, :, 0]
        else:
            gray = image.copy()

        # Handle categorical maps vs continuous images
        if set(np.unique(gray)).issubset({0, 1, 2}):
            binary = (gray > 0).astype(np.uint8) * 255
        else:
            # Check if dark-on-light or light-on-dark
            if gray.dtype != np.uint8:
                gray_u8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
            else:
                gray_u8 = gray

            # Try Otsu thresholding
            _, binary = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # If corners are predominantly white (light background), invert mask
            corner_sample = np.concatenate([
                binary[:5, :5].flatten(),
                binary[:5, -5:].flatten(),
                binary[-5:, :5].flatten(),
                binary[-5:, -5:].flatten(),
            ])
            if np.mean(corner_sample) > 128:
                binary = cv2.bitwise_not(binary)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        # Find largest contour matching circular/disc criteria
        candidate_contours = [c for c in contours if cv2.contourArea(c) > (0.05 * h * w)]
        if not candidate_contours:
            candidate_contours = contours

        largest_contour = max(candidate_contours, key=cv2.contourArea)
        (xc_f, yc_f), radius_f = cv2.minEnclosingCircle(largest_contour)
        xc, yc, radius = int(xc_f), int(yc_f), int(radius_f)

        # Sanity check radius bounds
        min_dim = min(h, w)
        if radius < min_dim * 0.15:
            # Fallback to center-crop if circle detection is degenerate
            half_side = min_dim // 2
            yc, xc = h // 2, w // 2
            radius = half_side

        # Add slight margin
        pad = int(radius * self.roi_padding_ratio)
        r_eff = radius + pad

        # Compute square bounding box
        y1 = max(0, yc - r_eff)
        y2 = min(h, yc + r_eff)
        x1 = max(0, xc - r_eff)
        x2 = min(w, xc + r_eff)

        crop = image[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        # Make square with edge padding if cropped against image boundary
        if ch != cw:
            target_dim = max(ch, cw)
            pad_top = (target_dim - ch) // 2
            pad_bottom = target_dim - ch - pad_top
            pad_left = (target_dim - cw) // 2
            pad_right = target_dim - cw - pad_left

            if crop.ndim == 3:
                padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            else:
                padding = ((pad_top, pad_bottom), (pad_left, pad_right))
            crop = np.pad(crop, padding, mode='constant', constant_values=0)

        return crop

    def process(self, image: np.ndarray) -> np.ndarray:
        """Processes raw 2D/3D wafer input into a normalized (target_size) array.

        Args:
            image: Raw image or 2D matrix (values 0: Background, 1: Good, 2: Defect, or uint8 image).

        Returns:
            Normalized 2D float32 or uint8 array resized to target_size.
        """
        # Auto-crop ROI if enabled
        if self.auto_crop_roi:
            image = self.extract_wafer_roi(image)

        # Convert RGB/RGBA to grayscale if 3D
        if image.ndim == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image[:, :, 0]
        else:
            gray = image.copy()

        # Check if already categorical {0, 1, 2}
        unique_vals = np.unique(gray)
        is_categorical = set(unique_vals).issubset({0, 1, 2})

        if is_categorical:
            # Categorical matrix: nearest neighbor interpolation to preserve discrete die labels
            resized = cv2.resize(
                gray.astype(np.uint8),
                self.target_size,
                interpolation=cv2.INTER_NEAREST,
            )
            return resized
        else:
            # Grayscale intensity image: bilinear resize + contrast normalization
            resized = cv2.resize(
                gray.astype(np.float32),
                self.target_size,
                interpolation=cv2.INTER_AREA,
            )
            # Min-max normalization
            min_val, max_val = np.min(resized), np.max(resized)
            if max_val > min_val:
                normalized = (resized - min_val) / (max_val - min_val) * 255.0
            else:
                normalized = resized
            return normalized.astype(np.uint8)
