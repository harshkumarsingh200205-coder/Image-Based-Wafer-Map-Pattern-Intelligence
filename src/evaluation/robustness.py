"""Controlled robustness and perturbation testing harness."""

from typing import Dict
import cv2
import numpy as np


def test_robustness_perturbations(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Applies controlled physical perturbations to evaluate model robustness.

    Perturbations:
    - Gaussian Noise
    - Small Rotation (+15 degrees)
    - Brightness / Contrast Scaling
    - Partial Radial Occlusion
    """
    h, w = image.shape[:2]
    perturbed_images = {}

    # 1. Noise injection
    noise = np.random.normal(0, 15, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    perturbed_images["gaussian_noise"] = noisy

    # 2. Small Rotation (+15 deg)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST)
    perturbed_images["rotation_15deg"] = rotated

    # 3. Contrast / Brightness shift
    contrast = np.clip(image.astype(np.float32) * 1.25 + 10, 0, 255).astype(np.uint8)
    perturbed_images["contrast_shift"] = contrast

    # 4. Partial edge occlusion
    occluded = image.copy()
    occluded[: h // 4, : w // 4] = 0
    perturbed_images["partial_occlusion"] = occluded

    return perturbed_images
