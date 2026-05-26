"""
Face feature extraction with ResNet-18.

WHAT AN "EMBEDDING" IS
----------------------
You cannot feed raw pixels straight into a simple classifier. Instead you turn
each image into a fixed-length list of numbers -- an "embedding" -- that
summarises what the image contains. A good embedding puts similar images close
together in number-space. ResNet-18 produces a 512-number embedding per image.

WHY RESNET-18, PRE-TRAINED, FROZEN
----------------------------------
- ResNet-18 is a small, well-understood image network.
- "Pre-trained" means it already learned general visual features from millions
  of everyday images (ImageNet). We reuse that knowledge instead of training
  from scratch -- essential when you only have ~100 patients.
- "Frozen" means we do NOT change its internal weights. We just run images
  through it and read off the embedding. This is fast, needs no GPU, and cannot
  overfit your small dataset. (Fine-tuning is a Block B sensitivity analysis,
  not the baseline.)

PER-PATIENT AGGREGATION
-----------------------
Each patient has several images (frontal, profiles, ...). We embed each image,
then combine that patient's embeddings into ONE per-patient vector by taking
the mean and the max across their images, and concatenating the two. Result:
a 1024-number vector per patient (512 mean + 512 max).

OUTPUT
------
A pandas DataFrame, one row per patient, columns: study_id, face_000 ...
face_1023. This is saved to data/processed/ so it is computed once, not every
time you train a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from airway import config, face_align

EMBED_DIM = 512                       # ResNet-18 embedding length per image
SEED = config.RANDOM_SEED

# torchvision's standard preprocessing for ImageNet-trained models:
# convert to tensor, then normalise each colour channel with ImageNet stats.
_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _build_embedder() -> nn.Module:
    """
    Load ResNet-18 pre-trained on ImageNet, with its final classification
    layer removed so it outputs the 512-number embedding.

    Returns a network in eval mode (no training behaviour).

    PRE-TRAINED WEIGHTS
    -------------------
    The first time this runs, torchvision downloads the ImageNet weights
    (~45 MB) and caches them. On a normal machine with internet this just
    works. If the download fails (no internet, or a restricted network), the
    code falls back to RANDOM weights so the pipeline still runs end-to-end --
    but a clear warning is printed, because random-weight embeddings are NOT
    suitable for real results. On your real machine, make sure the download
    succeeds before trusting any face-modality numbers.
    """
    try:
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
    except Exception as err:                       # download / network failure
        print(
            "  WARNING: could not download ImageNet weights "
            f"({type(err).__name__}). Falling back to RANDOM weights.\n"
            "  The pipeline will run, but face-modality results are NOT valid\n"
            "  until pre-trained weights load successfully on your machine."
        )
        net = models.resnet18(weights=None)        # random initialisation

    # ResNet-18's last layer (net.fc) maps 512 -> 1000 ImageNet classes.
    # Replace it with nn.Identity() so the network stops at the 512 embedding.
    net.fc = nn.Identity()

    net.eval()              # eval mode: disables dropout/batchnorm updates
    for param in net.parameters():
        param.requires_grad = False   # frozen: no weight changes
    return net


# Build the network once at import time and reuse it. The torch threads are
# pinned to 1 and the seed fixed so the embeddings are reproducible.
torch.manual_seed(SEED)
torch.set_num_threads(1)
_EMBEDDER = _build_embedder()


def _embed_one_image(image_path: str) -> np.ndarray:
    """Align one image and return its 512-number ResNet-18 embedding."""
    aligned = face_align.align_face(image_path)          # 224x224 RGB PIL image
    tensor = _PREPROCESS(aligned).unsqueeze(0)           # shape: (1, 3, 224, 224)
    with torch.no_grad():                                # no gradient bookkeeping
        emb = _EMBEDDER(tensor)                          # shape: (1, 512)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def extract_face_features(face_index: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the face image catalogue into a per-patient feature table.

    Parameters
    ----------
    face_index : DataFrame
        Output of loaders.face_loader(): columns study_id, view_code,
        file_path, abs_path. Several rows per patient.

    Returns
    -------
    DataFrame
        One row per patient. Columns: study_id, face_000 ... face_1023
        (512 mean-pooled + 512 max-pooled embedding values).

    Notes
    -----
    Progress is printed every 10 patients so you can see it is alive --
    embedding ~100 patients on CPU takes a few minutes, which is fine.
    """
    if config.ID_COL not in face_index.columns:
        raise ValueError(f"extract_face_features: need column '{config.ID_COL}'.")

    rows = []
    patient_ids = sorted(face_index[config.ID_COL].unique())

    for i, pid in enumerate(patient_ids, start=1):
        paths = face_index.loc[face_index[config.ID_COL] == pid, "abs_path"].tolist()

        per_image = []
        for path in paths:
            try:
                per_image.append(_embed_one_image(path))
            except (FileNotFoundError, ValueError) as err:
                # skip an unreadable image but keep going; report it
                print(f"  warning: skipping image for {pid}: {err}")

        if not per_image:
            # patient had zero usable images -- record NaNs, handle later
            print(f"  warning: {pid} has no usable images; row will be NaN.")
            combined = np.full(EMBED_DIM * 2, np.nan, dtype=np.float32)
        else:
            stack = np.vstack(per_image)                 # (n_images, 512)
            mean_vec = stack.mean(axis=0)                # (512,)
            max_vec = stack.max(axis=0)                  # (512,)
            combined = np.concatenate([mean_vec, max_vec])  # (1024,)

        row = {config.ID_COL: pid}
        for j, value in enumerate(combined):
            row[f"face_{j:03d}"] = value
        rows.append(row)

        if i % 10 == 0 or i == len(patient_ids):
            print(f"  embedded {i}/{len(patient_ids)} patients")

    return pd.DataFrame(rows)


def build_and_save_face_features() -> pd.DataFrame:
    """
    Run the full face feature extraction and save the result to
    data/processed/face_features.parquet.

    Returns the DataFrame as well, so callers can use it directly.
    """
    from airway import loaders

    config.ensure_dirs()
    face_index = loaders.face_loader()
    print(f"extract_face_features: {face_index[config.ID_COL].nunique()} patients, "
          f"{len(face_index)} images")

    features = extract_face_features(face_index)

    out = config.PROCESSED_DIR / "face_features.parquet"
    features.to_parquet(out, index=False)
    print(f"saved face features -> {out}  (shape {features.shape})")
    return features


if __name__ == "__main__":
    build_and_save_face_features()
