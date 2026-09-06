"""Grad-CAM visual activation heatmaps for deep learning model predictions."""

from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAMExplainer:
    """Computes gradient-weighted class activation maps (Grad-CAM) for CNN layers."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generates normalized 2D Grad-CAM heatmap overlay.

        Args:
            input_tensor: Tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses top predicted class.

        Returns:
            Normalized 2D float32 heatmap (0.0 to 1.0) resized to input spatial dims.
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients.detach().cpu().numpy()[0]  # (C, H', W')
        activations = self.activations.detach().cpu().numpy()[0]  # (C, H', W')

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU to keep only positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max

        # Resize to input dimensions
        _, _, h, w = input_tensor.shape
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam_resized
