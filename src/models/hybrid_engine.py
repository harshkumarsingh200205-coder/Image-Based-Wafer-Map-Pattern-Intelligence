"""Hybrid Decision Arbitration and Uncertainty Estimation Engine."""

from typing import Any, Dict, List, Optional
import numpy as np


class HybridDecisionEngine:
    """Arbitrates predictions between Classical Spatial heuristics, Feature-based ML, and Deep Learning CNN.

    Flags low-confidence, contradictory, or anomalous classifications as UNCERTAIN / NEEDS REVIEW.
    """

    def __init__(
        self,
        class_names: List[str],
        confidence_cutoff: float = 0.55,
        agreement_threshold: float = 0.65,
        uncertain_flag: str = "UNCERTAIN / NEEDS REVIEW",
    ):
        self.class_names = class_names
        self.confidence_cutoff = confidence_cutoff
        self.agreement_threshold = agreement_threshold
        self.uncertain_flag = uncertain_flag

    def arbitrate(
        self,
        ml_probs: np.ndarray,
        cnn_probs: np.ndarray,
        spatial_features: Dict[str, float],
    ) -> Dict[str, Any]:
        """Arbitrates final prediction across spatial features, ML, and CNN.

        Args:
            ml_probs: Class probability array from Classical ML (length = num_classes).
            cnn_probs: Class probability array from Deep CNN (length = num_classes).
            spatial_features: Dictionary of engineered spatial metrics.

        Returns:
            Dictionary containing final decision, confidence, agreement status, and model votes.
        """
        ml_idx = int(np.argmax(ml_probs))
        ml_conf = float(ml_probs[ml_idx])
        ml_label = self.class_names[ml_idx] if ml_idx < len(self.class_names) else "Unknown"

        cnn_idx = int(np.argmax(cnn_probs))
        cnn_conf = float(cnn_probs[cnn_idx])
        cnn_label = self.class_names[cnn_idx] if cnn_idx < len(self.class_names) else "Unknown"

        # Ensemble weighted probability
        ensemble_probs = 0.45 * ml_probs + 0.55 * cnn_probs
        top_idx = int(np.argmax(ensemble_probs))
        top_conf = float(ensemble_probs[top_idx])
        candidate_label = self.class_names[top_idx]

        is_agreement = (ml_label == cnn_label)
        has_high_confidence = (top_conf >= self.confidence_cutoff)

        # Spatial sanity verification rules
        spatial_contradiction = False
        center_to_edge = spatial_features.get("center_to_edge_ratio", 1.0)
        eccentricity = spatial_features.get("eccentricity", 0.0)

        if candidate_label == "Edge" and center_to_edge > 1.5:
            spatial_contradiction = True  # Edge claim contradicted by center concentration
        elif candidate_label == "Center" and center_to_edge < 0.4:
            spatial_contradiction = True  # Center claim contradicted by edge concentration
        elif candidate_label == "Scratch" and eccentricity < 0.45:
            spatial_contradiction = True  # Scratch claim contradicted by isotropic geometry

        # Final decision assignment
        if is_agreement and has_high_confidence and not spatial_contradiction:
            final_status = "CONSISTENT"
            final_prediction = candidate_label
            final_confidence = top_conf
        elif not is_agreement and top_conf >= 0.80 and not spatial_contradiction:
            # Strong single-model consensus overriding weaker model
            final_status = "RESOLVED_BY_MAJORITY"
            final_prediction = candidate_label
            final_confidence = top_conf * 0.85
        else:
            final_status = self.uncertain_flag
            final_prediction = self.uncertain_flag
            final_confidence = top_conf

        return {
            "final_prediction": final_prediction,
            "final_confidence": float(final_confidence),
            "status": final_status,
            "ml_prediction": ml_label,
            "ml_confidence": float(ml_conf),
            "cnn_prediction": cnn_label,
            "cnn_confidence": float(cnn_conf),
            "models_agree": bool(is_agreement),
            "spatial_contradiction": bool(spatial_contradiction),
        }
