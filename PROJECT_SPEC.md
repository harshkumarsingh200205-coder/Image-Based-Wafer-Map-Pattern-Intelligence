# Project Specification: Image-Based Wafer Map Pattern Intelligence

## 1. Problem Statement
In semiconductor manufacturing, silicon wafers undergo hundreds of chemical, photolithographic, and physical etching steps. When dies fail electrical testing, the spatial arrangement of failed dies across the wafer disk forms characteristic geometric patterns (e.g., center spots, edge rings, scratches, localized clusters). These spatial failure patterns directly correlate with specific fabrication equipment malfunctions (e.g., spin-coater nozzle clogging, CMP polishing non-uniformity, wafer handling scratch robot arms). 

Standard black-box image classifiers output simple class probabilities without explaining *why* a pattern was detected or verifying if the prediction makes physical, geometric sense. This project builds a transparent, multi-layered computer vision intelligence system for wafer map defect analysis.

## 2. Core Objectives
1. **End-to-End Visual Inspection Pipeline**: Take raw wafer-map images/arrays, validate integrity, segment wafer disc boundaries and defective die locations, and extract rigorous geometric/radial/density spatial statistics.
2. **Three-Layer Intelligence Architecture**:
   - **Layer A (Classical Computer Vision)**: Spatial and topological feature extraction (annular radial defect density, angular sector distribution, cluster compactness, eccentricity, center-to-edge ratio).
   - **Layer B (Machine Learning & Deep Learning)**: Benchmark spatial feature classifiers (Random Forest / SVM) against a lightweight Convolutional Neural Network (WaferNet).
   - **Layer C (Explainable Pattern Intelligence & Hybrid Decision)**: Grad-CAM visual activation heatmaps combined with rule-grounded spatial domain evidence, featuring an uncertainty-aware arbitration engine.
3. **Interactive Inspection Dashboard**: A local visual analytics application for engineering inspection, hypothesis testing, and quantitative evidence verification.

## 3. Supported Defect Pattern Classes
The system targets 9 canonical semiconductor wafer map failure signatures (adapted strictly to available dataset ground truth):
1. **Normal**: No statistically significant cluster or pattern; nominal yield.
2. **Center**: Defect dies concentrated in the innermost concentric zone.
3. **Donut**: Defect ring with a clean inner center and clean outer perimeter.
4. **Edge**: Defect dies concentrated on the outer circumference/bevel.
5. **Ring**: Continuous annular defect band located midway or near the perimeter.
6. **Localized Cluster**: Compact, high-density defect grouping in an arbitrary off-center location.
7. **Scratch**: Linear, high-aspect-ratio defect trajectory caused by mechanical handling tools.
8. **Random**: Uniformly scattered isolated defect dies with low spatial correlation.
9. **Mixed**: Superposition of multiple distinct pattern signatures (e.g., Ring + Scratch).

## 4. System Inputs & Outputs
- **Inputs**:
  - Raw wafer map images (PNG, JPG, BMP) or 2D die matrices (NumPy arrays / Pickle files representing 0: Background/Empty, 1: Good Die, 2: Defective Die).
- **Outputs**:
  - Preprocessed & wafer-masked image.
  - Defect die segmentation mask.
  - Quantitative spatial feature dictionary (Radial defect profile, Angular entropy, Center/Edge ratio, Cluster metrics).
  - Primary predicted class label and calibrated confidence score.
  - Decision agreement status: `CONSISTENT` or `UNCERTAIN / NEEDS REVIEW`.
  - Visual explanation: Grad-CAM activation overlay + Spatial rationale bullet points.

## 5. Scope Boundaries
- **In Scope**:
  - Classical CV wafer disk localization and defect die segmentation.
  - Mathematical extraction of spatial, radial, and density features.
  - Classical ML (Random Forest/SVM) and CNN (PyTorch) model training & benchmarking.
  - Grad-CAM and domain-specific spatial explanation generation.
  - Hybrid prediction arbitration and uncertainty estimation.
  - Controlled robustness testing (noise, rotation, scale, brightness).
  - Interactive Streamlit dashboard.
- **Out of Scope**:
  - Live hardware integration with factory wafer probers or fab SECS/GEM protocols.
  - Claims of industrial-grade production certification or wafer yield optimization guarantees.

## 6. Success & Verification Criteria
- [ ] Complete pipeline runs from raw image to multi-layer intelligence report without crashing.
- [ ] Strict data splitting with reproducible random seeds and data leakage guards.
- [ ] Macro-F1 score and confusion matrices generated across all classes.
- [ ] Uncertainty handling correctly identifies conflicting predictions between Classical CV and Deep Learning.
- [ ] Test coverage across preprocessing, segmentation, feature extraction, and hybrid engine.
