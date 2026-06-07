"""
Face alignment + normalisation -> persisted 224x224 eye-centred crops (Week 4).

WHAT THIS DOES
--------------
Turns each raw face photo into a clean, eye-aligned 224x224 RGB crop and SAVES
it to disk (data/processed/face_crops/). Saving matters: the crop is computed
once and reused by the embedding step, instead of being recomputed every run.

ALIGNMENT METHOD (with graceful fallback)
-----------------------------------------
Primary: dlib's frontal face detector + 68-point shape predictor. We read the
eye-corner landmarks, compute each eye centre, rotate the image so the eyes are
horizontal, then crop a square centred on the eye midpoint and resize to 224.

Fallback: if dlib or its 68-point model file (config.DLIB_LANDMARK_MODEL) is
unavailable, we fall back to OpenCV — Haar eye detection for eye-centred
alignment, or the face box / centre crop if eyes are not found. A clear message
is printed stating which path was used. Either way the pipeline runs and the
output is always a 224x224 RGB image.

To enable the dlib path:
    pip install dlib    # needs cmake + a C++ toolchain
    # download the model and place it at config.DLIB_LANDMARK_MODEL:
    #   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

IDEMPOTENCY
-----------
generate_crops() skips any image whose crop already exists on disk, unless
force=True. So re-running is cheap and safe.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from airway import config

TARGET_SIZE = 224
FACE_PAD = 0.35              # margin around the face box (fallback path)
EYE_TO_CENTER_SCALE = 2.2    # crop side = inter-eye distance * this (dlib path)

# Desired horizontal position of the eye midpoint in the output (centred).
_DESIRED_EYE_Y = 0.40        # eyes sit slightly above centre, like a passport crop


# ---------------------------------------------------------------------------
# Lazy singletons for the detectors, so import is cheap and failures are soft.
# ---------------------------------------------------------------------------
_DLIB = {"tried": False, "detector": None, "predictor": None}
_HAAR_FACE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
_HAAR_EYE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))


def _get_dlib():
    """Return (detector, predictor) if dlib + model are usable, else (None, None)."""
    if not _DLIB["tried"]:
        _DLIB["tried"] = True
        try:
            import dlib  # noqa: PLC0415
            if config.DLIB_LANDMARK_MODEL.exists():
                _DLIB["detector"] = dlib.get_frontal_face_detector()
                _DLIB["predictor"] = dlib.shape_predictor(str(config.DLIB_LANDMARK_MODEL))
            else:
                print(f"  face_crops: dlib present but model file missing at "
                      f"{config.DLIB_LANDMARK_MODEL}; using OpenCV fallback.")
        except Exception as err:  # pragma: no cover - environment dependent
            print(f"  face_crops: dlib unavailable ({type(err).__name__}); "
                  f"using OpenCV fallback.")
    return _DLIB["detector"], _DLIB["predictor"]


def alignment_backend() -> str:
    """Report which alignment path is active: 'dlib' or 'opencv'."""
    detector, predictor = _get_dlib()
    return "dlib" if (detector is not None and predictor is not None) else "opencv"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _align_by_eyes(bgr: np.ndarray, left_eye: tuple[float, float],
                   right_eye: tuple[float, float]) -> np.ndarray:
    """
    Rotate so the eyes are level, then crop a square centred on the eye midpoint.
    `left_eye`/`right_eye` are (x, y) in pixels (subject's left/right).
    """
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_dist = float(np.hypot(dx, dy))
    mid = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

    # rotate the whole image about the eye midpoint to make the eyes horizontal
    rot = cv2.getRotationMatrix2D(mid, angle, 1.0)
    rotated = cv2.warpAffine(bgr, rot, (bgr.shape[1], bgr.shape[0]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    side = max(eye_dist * EYE_TO_CENTER_SCALE, 16.0)
    x0 = int(round(mid[0] - side / 2.0))
    y0 = int(round(mid[1] - side * _DESIRED_EYE_Y))
    x1, y1 = int(round(x0 + side)), int(round(y0 + side))

    # pad if the crop runs off the edge, so output is always `side` x `side`
    pad = max(0, -x0, -y0, x1 - rotated.shape[1], y1 - rotated.shape[0])
    if pad:
        rotated = cv2.copyMakeBorder(rotated, pad, pad, pad, pad,
                                     cv2.BORDER_REFLECT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return rotated[y0:y1, x0:x1]


def _centre_square_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


# ---------------------------------------------------------------------------
# Per-backend eye finders
# ---------------------------------------------------------------------------
def _eyes_dlib(bgr: np.ndarray):
    detector, predictor = _get_dlib()
    if detector is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if not faces:
        return None
    face = max(faces, key=lambda r: r.width() * r.height())
    shape = predictor(gray, face)
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=float)
    # 68-point scheme: left eye = landmarks 36-41, right eye = 42-47
    left = pts[36:42].mean(axis=0)
    right = pts[42:48].mean(axis=0)
    return (float(left[0]), float(left[1])), (float(right[0]), float(right[1]))


def _eyes_opencv(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _HAAR_FACE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None, None
    fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    roi = gray[fy:fy + fh, fx:fx + fw]
    eyes = _HAAR_EYE.detectMultiScale(roi, 1.1, 5, minSize=(15, 15))
    if len(eyes) >= 2:
        # take the two largest eye boxes, order them left/right by x
        eyes = sorted(eyes, key=lambda b: b[2] * b[3], reverse=True)[:2]
        centres = [(fx + ex + ew / 2.0, fy + ey + eh / 2.0) for ex, ey, ew, eh in eyes]
        centres.sort(key=lambda c: c[0])
        return centres[0], centres[1]
    return (fx, fy, fw, fh), None   # signal: face box only, no eyes


def align_face(image_path: str | Path) -> Image.Image:
    """
    Load one raw image, return a 224x224 RGB eye-centred PIL image.

    Tries dlib 68-point landmarks first, then OpenCV eyes, then the face box,
    then a plain centre crop — whichever succeeds first. Never raises on a
    detect-miss (only on a missing/unreadable file).
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"align_face: no such image file: {image_path}")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"align_face: not a readable image: {image_path}")

    crop = None
    eyes = _eyes_dlib(bgr)
    if eyes is not None:
        crop = _align_by_eyes(bgr, eyes[0], eyes[1])
    else:
        left, right = _eyes_opencv(bgr)
        if left is not None and right is not None:
            crop = _align_by_eyes(bgr, left, right)
        elif left is not None:                       # face box, no eyes
            fx, fy, fw, fh = left
            cx, cy = fx + fw / 2, fy + fh / 2
            half = max(fw, fh) * (1 + FACE_PAD) / 2
            y0, y1 = int(max(0, cy - half)), int(min(bgr.shape[0], cy + half))
            x0, x1 = int(max(0, cx - half)), int(min(bgr.shape[1], cx + half))
            crop = bgr[y0:y1, x0:x1]

    if crop is None or crop.size == 0:
        crop = _centre_square_crop(bgr)

    resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def crop_path_for(study_id: str, view_code: str) -> Path:
    """Deterministic on-disk location for one patient/view crop."""
    return config.FACE_CROPS_DIR / f"{study_id}__{view_code}.png"


def generate_crops(force: bool = False, respect_quarantine: bool = True) -> dict:
    """
    Generate and persist aligned crops for every catalogued face image.

    Parameters
    ----------
    force : bool
        If False (default), skip images whose crop already exists -> idempotent.
        If True, recompute and overwrite every crop.
    respect_quarantine : bool
        If True and quarantine decisions exist, skip images excluded by the
        quarantine rules.

    Returns
    -------
    dict summary: counts of written / skipped / excluded / failed, and the
    active alignment backend.
    """
    from airway import loaders

    config.ensure_dirs()
    face_index = loaders.face_loader()

    excluded_paths: set[str] = set()
    if respect_quarantine:
        from airway import quarantine
        try:
            excluded_paths = quarantine.excluded_image_paths(quarantine.load_quarantine())
        except FileNotFoundError:
            pass  # no quarantine yet -> process everything

    backend = alignment_backend()
    print(f"generate_crops: backend={backend}, force={force}, "
          f"{len(face_index)} catalogued images")

    written = skipped = excluded = failed = 0
    for _, row in face_index.iterrows():
        if row["file_path"] in excluded_paths:
            excluded += 1
            continue
        out = crop_path_for(row[config.ID_COL], row["view_code"])
        if out.exists() and not force:
            skipped += 1
            continue
        try:
            img = align_face(row["abs_path"])
            img.save(out, format="PNG")
            written += 1
        except (FileNotFoundError, ValueError) as err:
            print(f"  warning: could not crop {row['abs_path']}: {err}")
            failed += 1

    summary = {"backend": backend, "written": written, "skipped": skipped,
               "excluded": excluded, "failed": failed,
               "total": len(face_index)}
    print(f"generate_crops: {summary}")
    return summary


def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Generate aligned 224x224 face crops.")
    p.add_argument("--force", action="store_true",
                   help="recompute crops even if they already exist")
    p.add_argument("--ignore-quarantine", action="store_true",
                   help="do not skip quarantine-excluded images")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    generate_crops(force=args.force, respect_quarantine=not args.ignore_quarantine)
