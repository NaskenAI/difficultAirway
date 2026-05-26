"""
Face alignment.

WHAT THIS DOES
--------------
A raw face photo has the face somewhere in the frame, at some size, against
some background. A neural network works best when every image is presented
the same way: face centred, same size, same pixel scale.

This module takes a raw image path and returns a clean, fixed-size
(224 x 224) RGB image ready for ResNet-18.

WHY 224 x 224
-------------
ResNet-18 was originally trained on 224 x 224 images. Feeding it that size
keeps everything consistent with how the network expects its input.

ABOUT "ALIGNMENT" IN THIS PILOT VERSION
---------------------------------------
True face alignment uses landmark detection (eyes, nose, mouth) to rotate and
crop precisely. That needs an extra model file (dlib's shape predictor) and
adds setup friction. For the pilot, this module does a robust, dependency-free
version: it detects the face with OpenCV's built-in face detector, takes a
square crop around it with padding, and resizes. If no face is detected, it
centre-crops the whole image instead, so the pipeline never crashes.

You can upgrade to landmark-based alignment later without changing anything
that calls this module -- the function signature stays the same.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = 224          # output is TARGET_SIZE x TARGET_SIZE
FACE_PAD = 0.35            # extra margin around the detected face box

# OpenCV ships a pre-trained face detector as an XML file. cv2.data.haarcascades
# is the folder where those files live inside the installed package.
_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(str(_CASCADE_PATH))


def _detect_face_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Return the (x, y, w, h) box of the largest detected face, or None.

    `gray` is a single-channel greyscale image (the detector wants greyscale).
    """
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    # if several faces are found, keep the biggest (largest w*h)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)


def _square_crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Take a padded SQUARE crop around `box`, clamped to the image edges."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + FACE_PAD)

    half = side / 2
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(img.shape[1], cx + half))
    y1 = int(min(img.shape[0], cy + half))
    return img[y0:y1, x0:x1]


def _centre_square_crop(img: np.ndarray) -> np.ndarray:
    """Fallback: take the largest centred square crop of the whole image."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def align_face(image_path: str | Path) -> Image.Image:
    """
    Load an image, locate the face, return a 224 x 224 RGB PIL image.

    Parameters
    ----------
    image_path : str or Path
        Path to a raw face image file (jpg/png).

    Returns
    -------
    PIL.Image.Image
        A 224 x 224 RGB image: face-centred if a face was found, otherwise a
        centre crop of the original.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the file exists but cannot be read as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"align_face: no such image file: {image_path}")

    # cv2.imread returns a numpy array in BGR channel order, or None on failure.
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(
            f"align_face: file exists but is not a readable image: {image_path}"
        )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    box = _detect_face_box(gray)

    crop = _square_crop(bgr, box) if box is not None else _centre_square_crop(bgr)

    # guard against a degenerate empty crop
    if crop.size == 0:
        crop = _centre_square_crop(bgr)

    # resize to the network's expected size
    resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    # convert BGR (OpenCV) -> RGB (PIL / torchvision expect RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def face_was_detected(image_path: str | Path) -> bool:
    """
    Convenience check: True if a face was detected in the image.

    Useful for the data-quality audit in Week 3 (what fraction of images had a
    detectable face). Does not raise on a bad file -- returns False instead.
    """
    try:
        bgr = cv2.imread(str(Path(image_path)))
        if bgr is None:
            return False
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return _detect_face_box(gray) is not None
    except Exception:
        return False
