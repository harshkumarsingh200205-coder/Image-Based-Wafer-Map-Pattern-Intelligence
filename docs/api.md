# Python API & Module Reference

## Preprocessing API (`src.preprocessing`)
- `ImageValidator.validate(image_input: Union[str, Path, np.ndarray]) -> Tuple[bool, Optional[str]]`: Validates image format, resolution, and pixel values.
- `WaferPreprocessor.process(image: np.ndarray, target_size=(128, 128)) -> np.ndarray`: Denoises, resizes, and normalizes wafer map representation.

## Segmentation API (`src.segmentation`)
- `WaferMaskDetector.detect(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]`: Returns binary wafer disc mask and `(center_x, center_y, radius)`.
- `DefectSegmenter.segment(image: np.ndarray, wafer_mask: np.ndarray) -> np.ndarray`: Returns binary defect die mask.

## Spatial Features API (`src.features`)
- `SpatialFeatureExtractor.extract(defect_mask: np.ndarray, wafer_mask: np.ndarray) -> Dict[str, Union[float, np.ndarray]]`: Extracts 25+ spatial, radial, and angular descriptors.

## Models & Hybrid Decision API (`src.models`)
- `HybridDecisionEngine.predict(image: np.ndarray, spatial_features: dict) -> Dict[str, Any]`: Executes multi-model inference and arbitrates certainty status.

## Explainability API (`src.explainability`)
- `GradCAMExplainer.generate(model, input_tensor: torch.Tensor, target_class: int) -> np.ndarray`: Generates Grad-CAM activation heatmap.
- `SpatialRationaleEngine.generate_rationale(spatial_features: dict, predicted_class: str) -> List[str]`: Generates domain evidence rationale statements.
