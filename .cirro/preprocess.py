#!/usr/bin/env python3
"""
Cirro preprocess script for hne-nuclear-segmentation pipeline.

Prepares the samplesheet.csv input file required by the nf-core/hnenucseg
Nextflow pipeline. Each input whole-slide image (WSI) file becomes one row
with columns `sample,slide_path`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from cirro.helpers.preprocess_dataset import PreprocessDataset

SAMPLESHEET_REQUIRED_COLUMNS = (
    "sample",
    "slide_path",
)

# Whole-slide image extensions supported via pyvips + OpenSlide.
# Lowercased; matched case-insensitively. `.ome.tif` / `.ome.tiff` are
# covered by the `.tif` / `.tiff` entries.
WSI_EXTENSIONS = frozenset(
    {
        ".svs",      # Aperio
        ".tif",      # generic / OME pyramidal TIFF
        ".tiff",
        ".ndpi",     # Hamamatsu
        ".mrxs",     # 3DHISTECH MIRAX
        ".vms",      # Hamamatsu
        ".vmu",      # Hamamatsu
        ".scn",      # Leica
        ".bif",      # Ventana
        ".czi",      # Zeiss
        ".qptiff",   # Akoya / PerkinElmer
        ".dcm",      # DICOM WSI
        ".isyntax",  # Philips
    }
)


def _is_wsi(file_path: str) -> bool:
    """Return True if `file_path` has a known whole-slide image extension."""
    p = Path(file_path)
    # Path.suffix is case-sensitive — Cirro file listings sometimes have
    # uppercase extensions (e.g. .SVS), so lowercase before comparing.
    return p.suffix.lower() in WSI_EXTENSIONS


def _resolve_slide_path(file_path: str) -> str:
    """Return the slide path. Path() collapses s3:// to s3:/, so revert it."""
    return str(Path(file_path)).replace("s3:/", "s3://")


def samplesheet_from_files(ds: PreprocessDataset) -> pd.DataFrame:
    """Build a samplesheet from Cirro's files DataFrame, filtering to WSI files."""
    files = ds.files

    ds.logger.info(f"Found {len(files)} files in ds.files")
    ds.logger.info(f"Columns: {list(files.columns)}")

    files = files.copy()

    # Filter to whole-slide image files only. Cirro datasets may include
    # sidecar files (e.g. .xml metadata, .json manifests) that we must skip.
    is_wsi = files["file"].apply(_is_wsi)
    skipped = files.loc[~is_wsi, "file"].tolist()
    if skipped:
        ds.logger.info(
            f"Skipping {len(skipped)} non-WSI file(s): {skipped}"
        )
    files = files.loc[is_wsi].copy()

    if files.empty:
        ds.logger.warning(
            "No WSI files found in dataset. "
            f"Supported extensions: {sorted(WSI_EXTENSIONS)}"
        )
        return pd.DataFrame(columns=list(SAMPLESHEET_REQUIRED_COLUMNS))

    files["slide_path"] = files["file"].apply(_resolve_slide_path)
    files = files[["sample", "slide_path"]]

    # If a sample has multiple slide files, keep one per sample. Users with
    # multi-slide samples should split them into distinct sample rows upstream.
    duplicates = files[files.duplicated(subset=["sample"], keep=False)]
    if not duplicates.empty:
        ds.logger.warning(
            f"Multiple slides per sample detected; keeping first. "
            f"Affected samples: {sorted(duplicates['sample'].unique().tolist())}"
        )
    files = files.drop_duplicates(subset=["sample"], keep="first")

    samplesheet = pd.merge(ds.samplesheet, files, on="sample", how="left")

    return samplesheet


def samplesheet_from_params(ds: PreprocessDataset) -> pd.DataFrame:
    """Fall back to ds.metadata['inputs'] when no files are found."""
    return pd.DataFrame(
        {
            "sample": [x["name"] for x in ds.metadata["inputs"]],
            "slide_path": [x["dataPath"] for x in ds.metadata["inputs"]],
        }
    )


def prepare_samplesheet(ds: PreprocessDataset) -> pd.DataFrame:
    """Prepare the samplesheet for the pipeline."""
    ds.logger.info(f"Params: {ds.params}")

    samplesheet = samplesheet_from_files(ds)

    if samplesheet.empty:
        ds.logger.warning(
            "No WSI files found in dataset. Preparing samplesheet from params."
        )
        samplesheet = samplesheet_from_params(ds)
        if samplesheet.empty:
            raise ValueError(
                "No WSI files found in dataset and unable to prepare "
                "samplesheet from params."
            )
        ds.logger.info("Prepared samplesheet from params.")

    for colname in SAMPLESHEET_REQUIRED_COLUMNS:
        if colname not in samplesheet.columns:
            ds.logger.warning(
                f"Samplesheet is missing required column '{colname}'. "
                "Populating with NaN."
            )
            samplesheet[colname] = np.nan

    # Drop rows with missing slide_path (samples that didn't match a WSI file)
    missing = samplesheet["slide_path"].isna()
    if missing.any():
        ds.logger.warning(
            f"Dropping {int(missing.sum())} sample(s) with no slide_path: "
            f"{samplesheet.loc[missing, 'sample'].tolist()}"
        )
        samplesheet = samplesheet.loc[~missing].copy()

    if samplesheet.empty:
        raise ValueError(
            "Samplesheet is empty after filtering to WSI files. "
            f"Supported extensions: {sorted(WSI_EXTENSIONS)}"
        )

    samplesheet.to_csv("samplesheet.csv", index=False)

    to_remove = []
    for k in ds.params:
        if k in SAMPLESHEET_REQUIRED_COLUMNS:
            ds.logger.info(
                f"Removing param '{k}' from dataset params as it is now "
                "in the samplesheet."
            )
            to_remove.append(k)

    for k in to_remove:
        ds.remove_param(k)

    ds.add_param("input", "samplesheet.csv", overwrite=True)

    ds.logger.info(f"Samplesheet:\n{samplesheet.to_string()}")

    return samplesheet


def main():
    ds = PreprocessDataset.from_running()

    ds.logger.info("Creating samplesheet from input files")
    ds.logger.info(f"Input files: {len(ds.files)} rows")
    ds.logger.info(f"Input columns: {list(ds.files.columns)}")
    ds.logger.info(f"Files DataFrame:\n{ds.files.to_string()}")

    samplesheet = prepare_samplesheet(ds)
    ds.logger.info(f"Samplesheet created with {len(samplesheet)} samples")

    ds.logger.info(f"Final params: {ds.params}")


if __name__ == "__main__":
    main()
