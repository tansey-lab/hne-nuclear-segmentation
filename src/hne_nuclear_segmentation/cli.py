"""Typer CLI."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from ._progress import setup_logging, step
from .io import read_geoparquet, write_geoparquet
from .slide import Slide

app = typer.Typer(add_completion=False, help="H&E nuclear segmentation pipeline.")


@app.callback()
def _root():
    setup_logging()


@app.command()
def tissue(
    slide_path: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(..., "--out"),
    thumbnail_mpp: float = typer.Option(8.0, "--thumbnail-mpp"),
    max_thumbnail_px: int = typer.Option(1024, "--max-thumbnail-px"),
    min_tissue_area_um2: float = typer.Option(5000.0, "--min-tissue-area-um2"),
    mpp: Optional[float] = typer.Option(None, "--mpp", help="Override base-level MPP if slide metadata lacks it"),
):
    """Detect tissue polygons and write GeoParquet."""
    from .tissue import detect_tissue

    slide = Slide.open(slide_path, mpp=mpp)
    gdf = detect_tissue(
        slide,
        thumbnail_mpp=thumbnail_mpp,
        max_thumbnail_px=max_thumbnail_px,
        min_tissue_area_um2=min_tissue_area_um2,
    )
    write_geoparquet(gdf, out)
    typer.echo(f"Wrote {len(gdf)} tissue polygons to {out}")


@app.command()
def tile(
    slide_path: Path = typer.Argument(..., exists=True, readable=True),
    tissue_path: Path = typer.Option(..., "--tissue", exists=True),
    out: Path = typer.Option(..., "--out"),
    target_mpp: float = typer.Option(0.5, "--target-mpp"),
    tile_size: int = typer.Option(1024, "--tile-size"),
    overlap: float = typer.Option(0.5, "--overlap", help="Tile stride = tile_size * (1 - overlap)."),
    edge_fraction: float = typer.Option(0.1, "--edge-fraction", help="Per-side inner keep-box inset, in tile-size units."),
    mpp: Optional[float] = typer.Option(None, "--mpp"),
):
    """Generate tiles intersecting tissue."""
    from .tiling import generate_tiles

    slide = Slide.open(slide_path, mpp=mpp)
    tissue_gdf = read_geoparquet(tissue_path)
    gdf = generate_tiles(
        slide,
        tissue_gdf,
        target_mpp=target_mpp,
        tile_size_model_px=tile_size,
        overlap_fraction=overlap,
        edge_fraction=edge_fraction,
    )
    write_geoparquet(gdf, out)
    typer.echo(f"Wrote {len(gdf)} tiles to {out}")


@app.command()
def infer(
    slide_path: Path = typer.Argument(..., exists=True, readable=True),
    tiles_path: Path = typer.Option(..., "--tiles", exists=True),
    model: str = typer.Option(..., "--model", help="stardist or cellpose"),
    out: Path = typer.Option(..., "--out"),
    batch_size: int = typer.Option(8, "--batch-size"),
    target_mpp: Optional[float] = typer.Option(None, "--target-mpp"),
    gpu: bool = typer.Option(True, "--gpu/--cpu"),
    mpp: Optional[float] = typer.Option(None, "--mpp"),
):
    """Run segmentation over tiles."""
    from .inference import run_inference

    slide = Slide.open(slide_path, mpp=mpp)
    tiles = read_geoparquet(tiles_path)

    if model == "stardist":
        from .models.stardist_model import StardistSegmenter

        seg = StardistSegmenter(target_mpp=target_mpp or 0.5)
    elif model == "cellpose":
        from .models.cellpose_model import CellposeSegmenter

        seg = CellposeSegmenter(target_mpp=target_mpp or 0.5, gpu=gpu, batch_size=batch_size)
    else:
        raise typer.BadParameter(f"Unknown model: {model}")

    gdf = run_inference(slide, tiles, seg, batch_size=batch_size, out_path=out)
    typer.echo(f"Wrote {len(gdf)} polygons to {out}")


@app.command()
def consensus(
    stardist: Path = typer.Option(..., "--stardist", exists=True),
    cellpose: Path = typer.Option(..., "--cellpose", exists=True),
    out_union: Path = typer.Option(..., "--out-union"),
    out_intersection: Path = typer.Option(..., "--out-intersection"),
    iou_threshold: float = typer.Option(0.3, "--iou-threshold"),
):
    """Build union + IoU-matched intersection consensus."""
    from .consensus import build_intersection, build_union, dedupe_overlapping

    s = dedupe_overlapping(read_geoparquet(stardist))
    c = dedupe_overlapping(read_geoparquet(cellpose))
    u = build_union(s, c)
    i = build_intersection(s, c, iou_threshold=iou_threshold)
    write_geoparquet(u, out_union)
    write_geoparquet(i, out_intersection)
    typer.echo(f"Union: {len(u)} | Intersection: {len(i)}")


@app.command()
def run(
    slide_path: Path = typer.Argument(..., exists=True, readable=True),
    out_dir: Path = typer.Option(..., "--out-dir"),
    target_mpp: float = typer.Option(0.5, "--target-mpp"),
    tile_size: int = typer.Option(1024, "--tile-size"),
    overlap: float = typer.Option(0.5, "--overlap", help="Tile stride = tile_size * (1 - overlap)."),
    edge_fraction: float = typer.Option(0.1, "--edge-fraction", help="Per-side inner keep-box inset, in tile-size units."),
    batch_size: int = typer.Option(8, "--batch-size"),
    iou_threshold: float = typer.Option(0.3, "--iou-threshold"),
    gpu: bool = typer.Option(True, "--gpu/--cpu"),
    mpp: Optional[float] = typer.Option(None, "--mpp", help="Override base-level MPP if slide metadata lacks it"),
    viz_html: bool = typer.Option(True, "--viz-html/--no-viz-html", help="Build interactive HTML viewer."),
    viz_pdf: bool = typer.Option(True, "--viz-pdf/--no-viz-pdf", help="Build full-res tile detail PDF."),
    viz_pdf_tiles: int = typer.Option(2, "--viz-pdf-tiles", help="Number of detail tiles for the PDF."),
):
    """End-to-end pipeline.

    Stardist (TensorFlow) and Cellpose (PyTorch) inference run in separate
    subprocesses. On macOS, loading both frameworks into one process pulls in
    two copies of libomp, which segfaults inside OpenMP worker creation when
    the second framework starts its thread pool. Subprocess isolation keeps
    each framework's runtime alone in its address space.
    """
    from .consensus import build_intersection, build_union, dedupe_overlapping
    from .tiling import generate_tiles
    from .tissue import detect_tissue

    out_dir.mkdir(parents=True, exist_ok=True)
    slide = Slide.open(slide_path, mpp=mpp)

    tiles_path = out_dir / "tiles.parquet"
    stardist_out = out_dir / "nuclei_stardist.parquet"
    cellpose_out = out_dir / "nuclei_cellpose.parquet"

    n_viz_steps = int(viz_html) + int(viz_pdf)
    total_steps = 5 + n_viz_steps

    with step(f"[1/{total_steps}] tissue detection"):
        tissue_gdf = detect_tissue(slide)
        write_geoparquet(tissue_gdf, out_dir / "tissue.parquet")

    with step(f"[2/{total_steps}] tiling"):
        tiles = generate_tiles(
            slide, tissue_gdf, target_mpp=target_mpp,
            tile_size_model_px=tile_size, overlap_fraction=overlap,
            edge_fraction=edge_fraction,
        )
        write_geoparquet(tiles, tiles_path)
        del tiles

    with step(f"[3/{total_steps}] stardist inference (subprocess)"):
        _run_infer_subprocess(
            slide_path=slide_path, tiles_path=tiles_path, model="stardist",
            out=stardist_out, batch_size=batch_size, target_mpp=target_mpp,
            gpu=gpu, mpp=mpp,
        )

    with step(f"[4/{total_steps}] cellpose inference (subprocess)"):
        _run_infer_subprocess(
            slide_path=slide_path, tiles_path=tiles_path, model="cellpose",
            out=cellpose_out, batch_size=batch_size, target_mpp=target_mpp,
            gpu=gpu, mpp=mpp,
        )

    with step(f"[5/{total_steps}] consensus"):
        s_gdf = dedupe_overlapping(read_geoparquet(stardist_out))
        c_gdf = dedupe_overlapping(read_geoparquet(cellpose_out))
        write_geoparquet(s_gdf, stardist_out)
        write_geoparquet(c_gdf, cellpose_out)
        write_geoparquet(build_union(s_gdf, c_gdf), out_dir / "consensus_union.parquet")
        write_geoparquet(
            build_intersection(s_gdf, c_gdf, iou_threshold=iou_threshold),
            out_dir / "consensus_intersection.parquet",
        )

    next_step = 6
    if viz_html:
        with step(f"[{next_step}/{total_steps}] interactive HTML viewer"):
            from .viz import build_html_viewer

            build_html_viewer(slide, out_dir)
        next_step += 1
    if viz_pdf:
        with step(f"[{next_step}/{total_steps}] detail PDF"):
            from .viz import build_detail_pdf

            build_detail_pdf(slide, out_dir, n_tiles=viz_pdf_tiles)

    typer.echo(f"Done. Outputs in {out_dir}")


@app.command()
def viz(
    slide_path: Path = typer.Argument(..., exists=True, readable=True),
    out_dir: Path = typer.Option(..., "--out-dir", exists=True, file_okay=False),
    html: bool = typer.Option(True, "--html/--no-html"),
    pdf: bool = typer.Option(True, "--pdf/--no-pdf"),
    pdf_tiles: int = typer.Option(2, "--pdf-tiles"),
    thumb_width: int = typer.Option(1800, "--thumb-width"),
    mpp: Optional[float] = typer.Option(None, "--mpp"),
):
    """Rebuild visualizations (HTML viewer, detail PDF) from existing pipeline outputs."""
    slide = Slide.open(slide_path, mpp=mpp)
    if html:
        from .viz import build_html_viewer

        build_html_viewer(slide, out_dir, thumb_width=thumb_width)
    if pdf:
        from .viz import build_detail_pdf

        build_detail_pdf(slide, out_dir, n_tiles=pdf_tiles)


def _run_infer_subprocess(
    *,
    slide_path: Path,
    tiles_path: Path,
    model: str,
    out: Path,
    batch_size: int,
    target_mpp: float,
    gpu: bool,
    mpp: Optional[float],
) -> None:
    cmd = [
        sys.executable, "-m", "hne_nuclear_segmentation.cli", "infer",
        str(slide_path),
        "--tiles", str(tiles_path),
        "--model", model,
        "--out", str(out),
        "--batch-size", str(batch_size),
        "--target-mpp", str(target_mpp),
        "--gpu" if gpu else "--cpu",
    ]
    if mpp is not None:
        cmd += ["--mpp", str(mpp)]

    env = os.environ.copy()
    if model == "cellpose":
        # PyTorch's bundled libomp clashes with other OMP runtimes loaded by
        # numpy/scipy/vips on macOS, crashing inside __kmp_create_worker.
        # Allow the duplicate load and serialize OMP so the runtimes don't
        # race each other while spinning up worker pools.
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    app()
