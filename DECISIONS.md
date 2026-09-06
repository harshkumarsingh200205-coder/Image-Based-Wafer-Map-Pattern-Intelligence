# Architecture Decision Records (ADRs)

## ADR-001: Three-Layer Intelligence vs. Pure End-to-End CNN
- **Context**: Deep learning classifiers can achieve high nominal accuracy on wafer images but often fail to provide verifiable physical justification or fail unpredictably when facing distribution shifts.
- **Decision**: Implement a 3-layer architecture (Classical CV $\rightarrow$ Multi-Model Predictive Layer $\rightarrow$ Explainability & Hybrid Arbitration).
- **Rationale**: Semiconductor yield engineers need physical metrics (e.g. radial defect density, cluster compactness) to map defects to specific equipment faults (CMP, spin-coating, robotic handler).
- **Status**: Approved.

## ADR-002: Spatial Feature Engineering with Concentric Annular & Polar Bins
- **Context**: Wafer map patterns possess strong rotational and radial symmetries (Center, Ring, Donut, Edge).
- **Decision**: Extract 8 concentric radial annular densities and 12 angular sector bins relative to the estimated wafer centroid.
- **Rationale**: Provides rotation-aware and scale-invariant geometric descriptors that directly distinguish isotropic patterns (Center/Ring) from anisotropic ones (Scratch/Edge sector).
- **Status**: Approved.

## ADR-003: Uncertainty-Aware Hybrid Arbitration
- **Context**: Disagreement between spatial geometric rules and deep CNN features indicates anomalous, complex, or low-confidence samples.
- **Decision**: When Classical ML and CNN predictions disagree, or top-1 confidence $< 0.55$, output `UNCERTAIN / NEEDS REVIEW` instead of forcing a low-confidence label.
- **Rationale**: Academic and industrial inspection systems must acknowledge ambiguity rather than hallucinating confidence.
- **Status**: Approved.

## ADR-004: Primary Evaluation Metric: Macro-F1 over Raw Accuracy
- **Context**: Real-world wafer datasets (like WM-811K) are heavily imbalanced (Normal/None class often represents $> 80\%$ of samples).
- **Decision**: Macro-averaged F1 score and per-class Recall are designated as the primary evaluation metrics.
- **Rationale**: Prevents a naive majority-class classifier from appearing high-performing.
- **Status**: Approved.
