# Reproducible landmark model

The optional model is a standardized ridge regressor that proposes four diagonal endpoints from the selected cross-section. It is an assistive landmark proposal only; it does not change the geometry-derived CI/CVA/CVAI values and has not been clinically validated.

## Current feature schema (v2)

Version `landmark-ridge-2.0.0` uses 24 radial features at 15-degree intervals (`radial_000` through `radial_345`). Each feature is computed as the intersection distance from the polygon-area centroid along a fixed ray, normalized by the maximum radius. This avoids assigning the feature value based on whichever STL vertices happen to fall into a bin, which made the old 12-bin signature sensitive to mesh tessellation density. If a strongly concave outline has no ray intersection, the extractor uses a flagged directional projection fallback; the count is exposed as `approximateDirections` / `radialFallbackBins` and should be reviewed before accepting ML suggestions.

Training targets remain the eight normalized x/y coordinates of the four diagonal endpoints:

- `d1_start_x`, `d1_start_y`, `d1_end_x`, `d1_end_y`
- `d2_start_x`, `d2_start_y`, `d2_end_x`, `d2_end_y`

Values are normalized around the contour profile center by its maximum radius and are expected in `[-2, 2]`.

## Annotation CSV

Use one row per anonymized scan and one scan per child. Do not include names, dates of birth, hospital IDs, or other identifying information. Required columns:

- `scan_code`: random, non-identifying scan code; also the grouping key for train/test separation.
- `radial_000` through `radial_345`: the 24 v2 radial features at 15-degree intervals.
- The eight expert-reviewed endpoint coordinate columns listed above.

Only expert-approved training scans belong in a clinical training CSV. The trainer makes a deterministic 80/20 split at the scan-code level, fits ridge regression, and compares held-out normalized-coordinate MAE with a mean-landmark baseline. The test scan codes and split sizes are recorded in the model artifact. A held-out result worse than baseline is explicitly warned about. With only 12 scans, a 24-feature model is underpowered; the minimum is a software guard, not a recommendation for clinical training.

## Train locally

```powershell
node ml\train-landmarks.mjs --input C:\private-data\landmark-labels.csv --seed 42 --lambda 0.1
```

By default, weights are written to `data/models/landmark-model.json`. The local server loads the model on its next start. Use `--output` to write elsewhere. The trainer also accepts legacy 12-column v1 CSV files and will produce a v1 artifact so earlier models remain supported. To use the 24-feature v2 model, regenerate the CSV with the v2 feature extractor and expert-reviewed targets.

## Technical-demo pseudo-labels (not clinical labels)

For a software demonstration only, a CSV can be generated from STL geometry:

```powershell
node ml\prepare-pseudo-labels.mjs --input-dir data\training-stl --output data\pseudo-landmarks.csv
node ml\train-landmarks.mjs --input data\pseudo-landmarks.csv --seed 42 --lambda 0.1
```

This script uses the deterministic geometry algorithm to create its own targets. It proves that feature extraction, training, model serialization and inference work end to end, but the model learns to imitate the geometry algorithm. Its held-out MAE is **not** clinical accuracy, independent validation, or expert agreement. Do not present it as a clinically trained model.

## Geometry section improvements

The section extractor now handles triangles crossed through vertices/edges and coplanar triangle patches, merges intersection points with a scale-aware spatial tolerance, removes duplicate segments and only accepts unambiguous degree-2 closed components. Diagnostics include segment/node counts, closed loops, branch/open nodes and the selected loop area. PCA orientation uses equal arc-length boundary samples to reduce bias from uneven triangle tessellation. The selected slice height is not silently moved, because doing so could change the anatomical measurement plane.

## Compatibility and validation

- v1 (`landmark-ridge-1.0.0`): legacy 12-bin inference remains supported.
- v2 (`landmark-ridge-2.0.0`): 24 ray/polygon features at 15-degree intervals.
- Geometry method version: `cranio-geometry-0.2.0`.

The tests verify deterministic parsing, section intersection robustness, feature stability under different contour tessellation densities, and reproducible training mechanics. They cannot establish clinical accuracy or the proposed 1.0 mm / percentage-point targets. Those require expert annotations and a pre-agreed independent validation protocol.
