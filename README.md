# Image-Based Wafer Map Pattern Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An end-to-end computer vision intelligence system for semiconductor wafer-map defect pattern recognition, spatial geometric analytics, and explainable decision auditing.

---

## 🔍 System Overview

Rather than relying on a naive "black-box" image classifier, this project integrates three complementary intelligence layers:

1. **Layer A — Classical Computer Vision & Spatial Geometry**:
   - Automated circular wafer disk detection and orientation normalization.
   - Adaptive die-level defect segmentation and connected-component analysis.
   - Concentric annular radial defect densities, angular polar sector profiles, center-to-edge ratios, and topological compactness metrics.
2. **Layer B — Multi-Model Predictive Layer**:
   - Feature-based machine learning (Random Forest / SVM) trained on physical spatial descriptors.
   - Lightweight Deep Convolutional Neural Network (`WaferNet_Light` in PyTorch).
3. **Layer C — Explainability & Uncertainty-Aware Hybrid Decision Engine**:
   - Grad-CAM activation heatmaps displaying salient wafer subregions.
   - Physical spatial evidence synthesis (e.g. radial concentration, defect centroid offset).
   - Conflict arbitration: Flags diverging or low-confidence samples as `UNCERTAIN / NEEDS REVIEW`.

---

## 📁 Repository Structure

```text
├── configs/             # Hyperparameters, paths, and model settings
│   └── config.yaml
├── data/                # Raw, processed, and split dataset artifacts
│   ├── raw/
│   ├── processed/
│   └── splits/
├── docs/                # Detailed technical and mathematical documentation
│   ├── setup.md
│   ├── pipeline.md
│   ├── spatial_features.md
│   └── testing.md
├── notebooks/           # Exploratory and prototyping notebooks
├── src/                 # Reusable production source code
│   ├── preprocessing/   # Ingestion, validation, filtering
│   ├── segmentation/    # Wafer mask & defect segmentation
│   ├── features/        # Radial, angular, and topological feature extractors
│   ├── models/          # Classical ML, CNN, and Hybrid arbitration
│   ├── explainability/  # Grad-CAM and domain spatial rationale
│   ├── evaluation/      # Metrics, ablation study, robustness
│   ├── visualization/   # Multi-stage wafer plotters and polar radars
│   └── utils/           # Logging and reproducibility seeds
├── app/                 # Interactive Streamlit analytics dashboard
│   └── app.py
├── tests/               # Pytest unit and integration test suite
├── reports/             # Generated figures, charts, and metric tables
├── results/             # Checkpoints, feature caches, predictions
├── requirements.txt     # Pinned Python dependencies
├── PROJECT_SPEC.md      # Detailed problem and scope specification
├── REQUIREMENTS.md      # Functional & non-functional requirements
├── ARCHITECTURE.md      # Architectural design and dataflow
├── DECISIONS.md         # Architecture Decision Records (ADRs)
└── TASKS.md             # Milestone progression and tracking
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/wafer-map-pattern-intelligence.git
cd wafer-map-pattern-intelligence

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard
```bash
streamlit run app/app.py
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 📊 Defect Pattern Signatures

| Pattern | Physical Characteristics | Spatial Indicator |
| :--- | :--- | :--- |
| **Normal** | Negligible defect count, uniform passing dies | Defect density $< 1\%$ |
| **Center** | High defect density concentrated near wafer center | Inner radial zone density $\gg$ Outer |
| **Donut** | Circular ring with clear inner center and clean outer rim | High mid-radius annular density |
| **Edge** | Defect concentration along the wafer perimeter | Outer radial zone density $\gg$ Inner |
| **Ring** | Continuous or segmented circle of defective dies | Concentric annular peak |
| **Localized Cluster** | Dense, compact grouping in an off-center region | High DBSCAN cluster density & low dispersion |
| **Scratch** | High aspect ratio, thin linear defect trajectory | High inertia ratio & eccentricity |
| **Random** | Uniformly scattered, uncorrelated defective dies | High spatial entropy, low clustering |
| **Mixed** | Superposition of multiple geometric defect signatures | Multi-modal radial/angular peaks |

---

## 📜 Academic Disclaimer
This project is developed as an academic computer vision and pattern recognition system (CSE3010-level). It is designed to demonstrate computer vision methodologies, spatial geometry feature engineering, and explainable decision auditing. It is an academic prototype and not intended for direct industrial fabrication tool control.
