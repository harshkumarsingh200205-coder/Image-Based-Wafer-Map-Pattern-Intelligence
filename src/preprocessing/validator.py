"""Input validation guards for wafer map images and matrices."""

from pathlib import Path
from typing import Tuple, Union, Optional
# pyrefly: ignore [missing-import]
import numpy as np


class ImageValidator:
    """Validates raw wafer image inputs to ensure structural integrity and bounds."""

    def __init__(
        self,
        min_resolution: Tuple[int, int] = (32, 32),
        max_aspect_ratio_diff: float = 0.35,
        auto_crop: bool = False,
    ):
        self.min_h, self.min_w = min_resolution
        self.max_aspect_diff = max_aspect_ratio_diff
        self.auto_crop = auto_crop

    def validate(
        self, image_input: Union[str, Path, np.ndarray]
    ) -> Tuple[bool, Optional[str]]:
        """Validates the input array or image path.

        Returns:
            Tuple (is_valid: bool, error_message: Optional[str])
        """
        # Array validation
        if isinstance(image_input, np.ndarray):
            arr = image_input
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return False, f"File does not exist: {path}"
            if path.stat().st_size == 0:
                return False, f"File is empty (zero bytes): {path}"
            try:
                import cv2
                arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if arr is None:
                    return False, f"Unsupported or corrupted image file: {path}"
            except Exception as e:
                return False, f"Failed to read image file: {str(e)}"
        else:
            return False, f"Unsupported input type: {type(image_input)}"

        # Dimension & Shape Checks
        if arr.ndim not in (2, 3):
            return False, f"Invalid array dimensions: {arr.ndim}. Expected 2D or 3D array."

        # If auto_crop is enabled, attempt ROI extraction first
        if self.auto_crop:
            from src.preprocessing.preprocessor import WaferPreprocessor
            arr = WaferPreprocessor().extract_wafer_roi(arr)

        h, w = arr.shape[:2]
        if h < self.min_h or w < self.min_w:
            return False, (
                f"Image resolution ({h}x{w}) is below the minimum required "
                f"({self.min_h}x{self.min_w})."
            )

        # Aspect ratio check
        aspect_ratio = max(h, w) / max(min(h, w), 1)
        if (aspect_ratio - 1.0) > self.max_aspect_diff:
            return False, (
                f"Wafer map aspect ratio is distorted ({aspect_ratio:.2f}). "
                "Silicon wafers must be approximately circular/square (1:1). "
                "Enable 'Auto-Crop Wafer Disc ROI' if uploading a screenshot or rectangular canvas."
            )

        # Entropy / Blank check
        if np.all(arr == arr.flat[0]):
            return False, "Image has zero variance (entirely blank/monochrome)."

        return True, None
