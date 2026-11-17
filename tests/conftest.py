import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def astronaut():
    """A real photo of a face (bundled with scikit-image) as a BGR image."""
    skimage = pytest.importorskip("skimage.data")
    return cv2.cvtColor(skimage.astronaut(), cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="session")
def cat():
    """A photo with no face in it."""
    skimage = pytest.importorskip("skimage.data")
    return cv2.cvtColor(skimage.chelsea(), cv2.COLOR_RGB2BGR)


def variants(img, n=6):
    """Slightly different versions of one photo: brightness, blur, small shifts."""
    out = []
    for i in range(n):
        v = cv2.convertScaleAbs(img, alpha=1.0, beta=(i - n // 2) * 8)
        if i % 2:
            v = cv2.GaussianBlur(v, (3, 3), 0)
        m = np.float32([[1, 0, i - n // 2], [0, 1, (i % 3) - 1]])
        out.append(cv2.warpAffine(v, m, (v.shape[1], v.shape[0]),
                                  borderMode=cv2.BORDER_REPLICATE))
    return out


@pytest.fixture
def eileen_dataset(tmp_path, astronaut):
    """dataset/eileen_collins/ with several real-face variants."""
    d = tmp_path / "dataset" / "Eileen Collins"
    d.mkdir(parents=True)
    for i, v in enumerate(variants(astronaut)):
        cv2.imwrite(str(d / f"{i}.png"), v)
    return d.parent


def studio_backgrounds(img):
    """The same face on a very dark and a very bright background (press photos)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (224, 114), (70, 90), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)[..., None]
    f = img.astype(np.float32)
    return {"dark_bg": (f * (mask + (1 - mask) * 0.15)).astype(np.uint8),
            "bright_bg": (f * mask + 255 * (1 - mask) * 0.9).astype(np.uint8)}
