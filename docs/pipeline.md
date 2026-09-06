# End-to-End Inspection Pipeline

The inspection pipeline transforms a raw wafer map image or array into an explainable pattern intelligence report across 5 distinct stages.

---

## Pipeline Execution Stages

### Stage 1: Validation (`src/preprocessing/validator.py`)
- Confirms input dimensions ($H \ge 32, W \ge 32$).
- Checks file integrity, numeric datatype bounds, and non-empty pixel entropy.
- Returns structured validation status: `(is_valid: bool, error_msg: Optional[str])`.

### Stage 2: Disc & Defect Segmentation (`src/segmentation/`)
- **Wafer Disc Localization (`wafer_mask.py`)**: Identifies circular wafer disc center $(x_0, y_0)$ and radius $R$. Clears off-wafer carrier pixels.
- **Defect Segmentation (`defect_segmenter.py`)**: Extracts the binary defect mask $M(x, y) \in \{0, 1\}$ representing failing dies.

### Stage 3: Spatial Feature Engineering (`src/features/`)
- Computes radial annular density vectors $\mathbf{d}_{\text{rad}} \in \mathbb{R}^8$.
- Computes polar angular density vectors $\mathbf{d}_{\text{ang}} \in \mathbb{R}^{12}$.
- Computes global spatial metrics (Center-to-Edge ratio, defect eccentricity, cluster compactness, inertia ratio).

### Stage 4: Multi-Model Inference & Hybrid Arbitration (`src/models/`)
- **Classical Model**: Predicts class probability vector using Random Forest over spatial feature vector.
- **Deep CNN**: Predicts class probability vector from normalized wafer tensor.
- **Hybrid Engine**: Evaluates prediction agreement and confidence. Assigns consensus prediction or `UNCERTAIN / NEEDS REVIEW`.

### Stage 5: Explainability & Reporting (`src/explainability/`)
- **Grad-CAM**: Computes backpropagated convolutional gradients over the salient wafer regions.
- **Spatial Rationale**: Translates quantitative spatial feature distributions into clear, human-readable evidence statements.
