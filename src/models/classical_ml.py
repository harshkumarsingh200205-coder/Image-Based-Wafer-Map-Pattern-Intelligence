"""Classical feature-based machine learning model (Random Forest / Gradient Boosting / SVM)."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .base import BaseWaferClassifier


class ClassicalSpatialClassifier(BaseWaferClassifier):
    """Classifies wafer patterns from engineered spatial/radial feature vectors."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fits scaler and Random Forest model on extracted spatial features."""
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.classes_ = self.clf.classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels."""
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)
