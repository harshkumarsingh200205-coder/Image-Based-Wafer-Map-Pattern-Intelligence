"""Abstract base interface for wafer map classifiers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np


class BaseWaferClassifier(ABC):
    """Abstract interface defining the contract for all pattern classifiers."""

    @abstractmethod
    def fit(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> None:
        """Trains the classifier on labeled wafer representations."""
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, List]) -> np.ndarray:
        """Predicts class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, List]) -> np.ndarray:
        """Predicts class probability distributions."""
        pass
