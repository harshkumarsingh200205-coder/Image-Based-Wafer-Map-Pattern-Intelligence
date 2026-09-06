# System Requirements Specification

## 1. Functional Requirements (FR)

### FR-001: Image Ingestion & Input Validation
- The system MUST accept 2D array inputs and standard image formats (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.npy`, `.pkl`).
- The system MUST validate image resolution, aspect ratio, channel integrity, and non-empty pixel distribution before processing.
- The system MUST reject corrupted or out-of-spec inputs with descriptive error messages without crashing.

### FR-002: Preprocessing & Wafer Disc Segmentation
- The system MUST isolate the active circular wafer disc from the surrounding background using Hough circle transform or contour thresholding.
- The system MUST normalize wafer orientation and scale to a canonical coordinate frame (default: 128x128).

### FR-003: Defect Die Segmentation
- The system MUST segment defective dies from passing dies using adaptive thresholding and morphological operations.
- The system MUST output a binary defect mask `(H, W)` where `1` represents a defective die and `0` represents background/passing die.

### FR-004: Spatial & Geometric Feature Engineering
- The system MUST compute annular radial defect densities across concentric zones (default: 8 radial bins from center $r=0$ to circumference $r=R$).
- The system MUST compute angular sector distributions (default: 12 directional bins).
- The system MUST compute center-to-edge defect ratios, global defect density, cluster compactness, inertia ratio, convex hull solidity, and connected-component metrics.

### FR-005: Multi-Model Pattern Recognition
- **Subsystem A (Classical ML)**: The system MUST provide a feature-based classifier (Random Forest / SVM) trained on the extracted spatial feature vectors.
- **Subsystem B (Deep Learning)**: The system MUST provide a lightweight CNN model (WaferNet) accepting normalized wafer map images.
- The system MUST output predicted class probabilities and top-1 confidence scores for each model.

### FR-006: Hybrid Decision Arbitration & Uncertainty Estimation
- The system MUST compare predictions from Classical Spatial analysis, Feature-based ML, and Deep Learning CNN.
- When model predictions agree with high confidence, the system MUST emit the consensus classification.
- When model predictions diverge significantly or confidence falls below threshold ($< 0.55$), the system MUST assign the status `UNCERTAIN / NEEDS REVIEW`.

### FR-007: Explainable Pattern Intelligence
- The system MUST generate Grad-CAM visual heatmaps for CNN activations, highlighting the wafer subregions influencing the prediction.
- The system MUST generate a domain-specific textual rationale summarizing physical spatial evidence (e.g., "92% of defects reside in the outer 15% radial zone; Center defect density is 0.01").

### FR-008: Evaluation & Ablation Harness
- The system MUST compute multi-class Accuracy, Precision, Recall, Macro-F1, and Confusion Matrices.
- The system MUST support progressive ablation experiments comparing Raw Classifier vs Preprocessed vs Spatial ML vs CNN vs Hybrid Engine.

### FR-009: Robustness Testing Harness
- The system MUST evaluate model resilience under controlled perturbations: Gaussian noise, rotation ($\pm 15^\circ$), resizing, and contrast shifts.

### FR-010: Interactive Inspection Dashboard
- The system MUST provide a local Streamlit web dashboard allowing wafer upload, multi-stage pipeline inspection, interactive feature visualization, and explainability heatmaps.

---

## 2. Non-Functional Requirements (NFR)

### NFR-001: Performance & Latency
- Image preprocessing, feature extraction, and model inference MUST execute within $< 300\text{ ms}$ per wafer map on standard CPU hardware.

### NFR-002: Reproducibility & Determinism
- All data splitting, model training, and feature extraction pipelines MUST utilize explicit random seeds (`seed=42`).

### NFR-003: Modularity & Code Hygiene
- Code MUST follow clean object-oriented and functional modular separation (`src/preprocessing/`, `src/segmentation/`, `src/features/`, `src/models/`, `src/explainability/`, `src/evaluation/`, `src/visualization/`).
- Every module MUST include type hints, docstrings, and zero circular dependencies.

### NFR-004: Robust Error Handling
- Invalid, corrupted, or non-wafer images MUST produce structured error responses rather than unhandled Python exceptions.

---

## 3. Acceptance Verification Checklist
- [ ] Image validator catches corrupted and non-wafer inputs.
- [ ] Wafer mask and defect segmenter accurately extract circular disk and defect coordinates.
- [ ] Spatial feature extractor outputs deterministic feature vectors of fixed dimension.
- [ ] Classical ML and CNN models train and save checkpoint artifacts.
- [ ] Hybrid engine flags conflict test cases as `UNCERTAIN / NEEDS REVIEW`.
- [ ] Grad-CAM and spatial rationale generate visual/textual explanations.
- [ ] Streamlit dashboard runs and renders all 5 pipeline stages.
- [ ] Pytest suite passes 100%.
