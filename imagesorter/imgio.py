"""Image loading that also works on Windows paths with non-ASCII characters
(accents, Arabic, ...), where plain cv2.imread silently returns None."""
import cv2
import numpy as np


def read_image(path):
    """BGR image, or None if the file is missing, empty or not a valid image.
    JPEG EXIF rotation is applied, same as cv2.imread."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
