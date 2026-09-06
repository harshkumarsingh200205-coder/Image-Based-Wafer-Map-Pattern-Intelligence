# AI Development Rules: Wafer Map Pattern Intelligence

## 1. Pedagogical & Tutor Role
- **Teach Before Coding**: Explain the physical semiconductor and computer-vision concepts (e.g. why annular radial profiles distinguish Ring from Donut patterns) alongside code implementation.
- **Incremental Steps**: Build one modular component at a time. Never generate monolithic unverified scripts.
- **Explain Trade-offs**: Clarify trade-offs between classical geometric heuristics vs deep convolutional feature representations.

## 2. "Assume a Stranger's PR" Review Protocol
When verifying or modifying code, critically assess:
- **Correctness & Numerics**: Check for divide-by-zero errors in radial density ratios when wafers have 0 defect dies.
- **Data Leakage**: Ensure normalization statistics, spatial scaler fits, and train/val/test splits strictly prevent test contamination.
- **Reproducibility**: Ensure all operations use explicit random seeds (`seed=42`).
- **Input Validation**: Ensure every pipeline entry point validates image size, format, and numerical bounds.

## 3. Scope & Integrity Constraints
- Do not fabricate benchmark scores or accuracy figures. All reported metrics must originate from actual execution logs.
- Preserve clear separation between academic research experiments (`notebooks/`, `reports/`) and reusable modular code (`src/`).
