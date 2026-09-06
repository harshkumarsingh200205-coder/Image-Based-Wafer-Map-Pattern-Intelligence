# Testing Strategy & Quality Assurance

## 1. Automated Test Suites

The test suite in [`tests/`](file:///c:/Projects/Image-Based%20Wafer%20Map%20Pattern%20Intelligence/tests/) provides rigorous verification across all system components:

1. **`test_validator.py`**:
   - Asserts valid ingestion of normalized numpy matrices and synthetic wafer images.
   - Tests boundary conditions: corrupted images, non-square aspect ratios, extreme small images ($< 16\times16$), and all-zero/blank matrices.
2. **`test_segmentation.py`**:
   - Generates synthetic wafer disks with ground-truth radii and tests Hough circle / mask localization precision.
   - Tests defect die segmentation on clear vs noisy background conditions.
3. **`test_features.py`**:
   - Verifies that a synthetic Center defect pattern yields high inner radial density and low outer density.
   - Verifies that a synthetic Edge defect pattern yields high outer radial density.
   - Verifies that a linear Scratch pattern yields eccentricity $> 0.85$.
4. **`test_hybrid_engine.py`**:
   - Asserts that model agreement generates high confidence and `CONSISTENT` status.
   - Asserts that conflicting model predictions trigger `UNCERTAIN / NEEDS REVIEW`.

## 2. Running Tests
```bash
# Run complete test suite
pytest tests/ -v

# Run with test coverage report
pytest --cov=src tests/ --cov-report=term-missing
```
