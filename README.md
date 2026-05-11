# hne-nuclear-segmentation

Nuclear segmentation pipeline for H&E whole-slide images. Wraps **Stardist** (`2D_versatile_he`) and **Cellpose-SAM**, runs them over tissue-restricted tiles with edge-effect-aware overlap, and emits per-model plus consensus (union and IoU-matched intersection) segmentations as **GeoParquet** in WSI base-level pixel coordinates.

## Pipeline

```
slide (WSI) ─► tissue detection (Otsu on saturation)
              ─► tiling (within tissue, with overlap)
              ─► Stardist inference  ─┐
              ─► Cellpose inference  ─┤
                                       ├─► consensus union
                                       └─► consensus intersection (IoU-matched)
```

Edge artifacts are handled by extracting tiles with N% overlap and discarding any detected nucleus whose centroid does not fall inside the inner `(1 - 2·overlap)` keep-box of its source tile. This makes tile borders invisible in the final output without explicit cross-tile deduplication.

## Install

Requires Python 3.11 (Stardist requires TensorFlow which does not yet support 3.13).

```bash
uv venv --python 3.11
uv sync
```

GPU notes:
- Stardist uses TensorFlow. For NVIDIA GPUs install the matching `tensorflow[and-cuda]` build separately.
- Cellpose uses PyTorch. Install the matching CUDA wheel from pytorch.org if not picked up.

## Usage

End-to-end:

```bash
hne-segment run slide.svs --out-dir ./out --target-mpp 0.5 --tile-size 1024 --overlap 0.10
```

Per stage (all intermediate artifacts are GeoParquet):

```bash
hne-segment tissue    slide.svs --out out/tissue.parquet
hne-segment tile      slide.svs --tissue out/tissue.parquet --out out/tiles.parquet
hne-segment infer     slide.svs --tiles out/tiles.parquet --model stardist --out out/nuclei_stardist.parquet
hne-segment infer     slide.svs --tiles out/tiles.parquet --model cellpose --out out/nuclei_cellpose.parquet
hne-segment consensus --stardist out/nuclei_stardist.parquet --cellpose out/nuclei_cellpose.parquet \
                      --out-union out/consensus_union.parquet --out-intersection out/consensus_intersection.parquet
```

All output polygons are in **level-0 (base) pixel coordinates** of the input WSI.

## Outputs

| File | Geometry | Notes |
|------|----------|-------|
| `tissue.parquet` | Tissue regions | One row per tissue piece |
| `tiles.parquet` | Tile boxes | Includes `keep_x0..keep_y1` inner box |
| `nuclei_stardist.parquet` | Per-nucleus polygons | `model_name="stardist"` |
| `nuclei_cellpose.parquet` | Per-nucleus polygons | `model_name="cellpose"` |
| `consensus_union.parquet` | Union of overlapping detections | `models` lists contributors |
| `consensus_intersection.parquet` | Pair-matched (IoU ≥ threshold) | `iou` per row |
| `visualization.html` | Self-contained interactive viewer | Slide thumbnail + toggleable SVG polygon layers, pan/zoom |
| `details.pdf` | Full-resolution tile crops with overlays | N densest tiles rendered via matplotlib |

Visualizations are produced by default at the end of `hne-segment run`. Disable
with `--no-viz-html` / `--no-viz-pdf`, or rebuild later via
`hne-segment viz <slide> --out-dir <dir>`.
