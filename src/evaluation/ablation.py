"""Progressive 5-Stage Ablation Study Harness (Prompt Section 8).

Compares:
1. Exp 1: Raw Image Pixel Flattened Classifier (Baseline)
2. Exp 2: Preprocessed Image + Basic Classifier
3. Exp 3: Spatial & Radial Features + Classical ML (Random Forest)
4. Exp 4: Deep Learning CNN (WaferNet_Light)
5. Exp 5: Hybrid Spatial + ML/DL Decision Engine
"""

from typing import Dict, List
import pandas as pd


def run_ablation_benchmark(experiment_results: List[Dict[str, str]]) -> pd.DataFrame:
    """Formats and summarizes ablation benchmark comparison across the 5 architectures.

    Args:
        experiment_results: List of dicts with keys ['Approach', 'Accuracy', 'Macro_F1', 'Strength', 'Limitation']

    Returns:
        Structured comparison DataFrame.
    """
    df = pd.DataFrame(experiment_results)
    return df
