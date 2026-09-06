"""Dataset ingestion and stratified train/val/test splitting pipeline."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class WaferDatasetLoader:
    """Manages wafer dataset ingestion, partitioning, and leakage-safe splits."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        splits_dir: str = "data/splits",
        seed: int = 42,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.splits_dir = Path(splits_dir)
        self.seed = seed

    def create_stratified_splits(
        self,
        samples: List[Dict],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Creates leak-proof train/validation/test splits with stratification."""
        df = pd.DataFrame(samples)
        if "label" not in df.columns:
            raise ValueError("Dataset entries must contain a 'label' column.")

        # Stratified first split: train vs temp (val + test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=df["label"],
            random_state=self.seed,
        )

        # Stratified second split: val vs test
        relative_test_ratio = test_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            stratify=temp_df["label"],
            random_state=self.seed,
        )

        self.splits_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self.splits_dir / "train.csv", index=False)
        val_df.to_csv(self.splits_dir / "val.csv", index=False)
        test_df.to_csv(self.splits_dir / "test.csv", index=False)

        return train_df, val_df, test_df
