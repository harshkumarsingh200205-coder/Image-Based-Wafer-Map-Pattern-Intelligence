"""Unit tests for the Hybrid Decision Arbitration Engine."""
# pyrefly: ignore [missing-import]
import numpy as np
from src.models.hybrid_engine import HybridDecisionEngine


CLASSES = [
    "Normal",
    "Center",
    "Donut",
    "Edge",
    "Ring",
    "Localized_Cluster",
    "Scratch",
    "Random",
    "Mixed",
]


def test_hybrid_engine_agreement():
    engine = HybridDecisionEngine(class_names=CLASSES)

    # Both models predict Center with high probability
    ml_probs = np.zeros(len(CLASSES))
    ml_probs[CLASSES.index("Center")] = 0.92
    cnn_probs = np.zeros(len(CLASSES))
    cnn_probs[CLASSES.index("Center")] = 0.88

    spatial_features = {
        "center_to_edge_ratio": 3.5,
        "eccentricity": 0.1,
    }

    result = engine.arbitrate(ml_probs, cnn_probs, spatial_features)

    assert result["status"] == "CONSISTENT"
    assert result["final_prediction"] == "Center"
    assert result["models_agree"] is True
    assert result["final_confidence"] > 0.85


def test_hybrid_engine_conflict_triggers_uncertainty():
    engine = HybridDecisionEngine(class_names=CLASSES)

    # ML predicts Edge, but CNN predicts Scratch with conflicting moderate probabilities
    ml_probs = np.zeros(len(CLASSES))
    ml_probs[CLASSES.index("Edge")] = 0.52
    ml_probs[CLASSES.index("Center")] = 0.48

    cnn_probs = np.zeros(len(CLASSES))
    cnn_probs[CLASSES.index("Scratch")] = 0.54
    cnn_probs[CLASSES.index("Edge")] = 0.46

    spatial_features = {
        "center_to_edge_ratio": 1.0,
        "eccentricity": 0.5,
    }

    result = engine.arbitrate(ml_probs, cnn_probs, spatial_features)

    assert result["status"] == "UNCERTAIN / NEEDS REVIEW"
    assert result["final_prediction"] == "UNCERTAIN / NEEDS REVIEW"
    assert result["models_agree"] is False
