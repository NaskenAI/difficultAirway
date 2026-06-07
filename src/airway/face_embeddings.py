"""
Face embeddings (Week 5) + per-patient aggregation (Week 6 of the task list).

PIPELINE POSITION
-----------------
    raw images --(face_crops)--> 224x224 crops --(THIS FILE)--> 512-d per image
              --(aggregate)--> 1024-d per patient --> face_model

WHAT THIS DOES
--------------
1. Loads a frozen, ImageNet-pretrained ResNet-18 (reused from face_features so
   there is exactly ONE embedder definition in the project) and extracts a
   global-pooled 512-d embedding per face crop.
2. Persists the per-image embeddings to data/processed/face_embeddings.parquet
   (one row per patient/view). Computed ONCE, not per CV fold.
3. Aggregates a patient's image embeddings into a single 1024-d vector by
   concatenating the mean-pooled and max-pooled 512-d vectors, and saves it to
   data/processed/face_features.parquet (one row per patient).

WHY PERSIST PER-IMAGE EMBEDDINGS
--------------------------------
The embeddings are produced by a FROZEN network — they do not depend on the
train/test split, so computing them once outside cross-validation is correct
and leakage-free. Persisting them means the (slow) ResNet pass runs once; model
training then reads the parquet instantly inside each fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from airway import config, face_crops
from airway.face_features import _EMBEDDER, _PREPROCESS, EMBED_DIM

# (face_features defines the single frozen ResNet-18 embedder + preprocessing;
#  we reuse it so there is exactly one embedder definition in the project.)

IMG_PREFIX = config.FACE_IMG_EMBED_PREFIX    # per-image embedding cols: emb_000 .. emb_511
FACE_PREFIX = config.FACE_FEATURE_PREFIX     # per-patient feature cols: face_000 .. face_1023


def embed_pil(image) -> np.ndarray:
    """Run one already-aligned 224x224 RGB PIL image through ResNet-18 -> 512-d."""
    tensor = _PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        emb = _EMBEDDER(tensor)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def _embed_one(study_id: str, view_code: str, abs_path: str) -> np.ndarray:
    """
    Embed a single patient/view. Prefer the persisted crop; if it is not on
    disk, align on the fly from the raw image (does not persist it here).
    """
    crop_file = face_crops.crop_path_for(study_id, view_code)
    if crop_file.exists():
        from PIL import Image
        img = Image.open(crop_file).convert("RGB")
    else:
        img = face_crops.align_face(abs_path)
    return embed_pil(img)


def build_image_embeddings(force: bool = False,
                           respect_quarantine: bool = True) -> pd.DataFrame:
    """
    Compute and persist per-image 512-d embeddings.

    Idempotent: if the parquet already exists and force=False, it is loaded and
    returned without recomputing.
    """
    from airway import loaders

    config.ensure_dirs()
    out = config.FACE_EMBEDDINGS_PARQUET
    if out.exists() and not force:
        print(f"build_image_embeddings: using cached {out} (force=True to redo)")
        return pd.read_parquet(out)

    face_index = loaders.face_loader()

    excluded_paths: set[str] = set()
    if respect_quarantine:
        from airway import quarantine
        try:
            excluded_paths = quarantine.excluded_image_paths(quarantine.load_quarantine())
        except FileNotFoundError:
            pass

    rows = []
    n = len(face_index)
    for i, (_, row) in enumerate(face_index.iterrows(), start=1):
        if row["file_path"] in excluded_paths:
            continue
        try:
            emb = _embed_one(row[config.ID_COL], row["view_code"], row["abs_path"])
        except (FileNotFoundError, ValueError) as err:
            print(f"  warning: skipping {row['abs_path']}: {err}")
            continue
        rec = {config.ID_COL: row[config.ID_COL], "view_code": row["view_code"],
               "file_path": row["file_path"]}
        for j, v in enumerate(emb):
            rec[f"{IMG_PREFIX}{j:03d}"] = v
        rows.append(rec)
        if i % 20 == 0 or i == n:
            print(f"  embedded {i}/{n} images")

    emb_df = pd.DataFrame(rows)
    emb_df.to_parquet(out, index=False)
    print(f"saved per-image embeddings -> {out}  (shape {emb_df.shape})")
    return emb_df


def aggregate_per_patient(emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-pool and max-pool a patient's per-image embeddings and concatenate ->
    one 1024-d vector per patient (512 mean + 512 max).
    """
    emb_cols = [c for c in emb_df.columns if c.startswith(IMG_PREFIX)]
    if len(emb_cols) != EMBED_DIM:
        raise ValueError(
            f"aggregate_per_patient: expected {EMBED_DIM} embedding columns, "
            f"found {len(emb_cols)}.")

    rows = []
    for pid, grp in emb_df.groupby(config.ID_COL, sort=True):
        stack = grp[emb_cols].to_numpy(dtype=np.float32)   # (n_images, 512)
        combined = np.concatenate([stack.mean(axis=0), stack.max(axis=0)])  # (1024,)
        rec = {config.ID_COL: pid}
        for j, v in enumerate(combined):
            rec[f"{FACE_PREFIX}{j:03d}"] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def build_and_save_patient_features(force: bool = False) -> pd.DataFrame:
    """Full Week-5/6 step: per-image embeddings -> per-patient 1024-d features."""
    emb_df = build_image_embeddings(force=force)
    features = aggregate_per_patient(emb_df)
    out = config.FACE_FEATURES_PARQUET
    features.to_parquet(out, index=False)
    print(f"saved per-patient face features -> {out}  (shape {features.shape})")
    return features


def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Extract 512-d face embeddings and 1024-d patient features.")
    p.add_argument("--force", action="store_true",
                   help="recompute embeddings even if the parquet exists")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    build_and_save_patient_features(force=args.force)
