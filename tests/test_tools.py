"""Dataset audit tool: finds unusable photos and train/val leaks."""
import shutil
import sys

import cv2

from conftest import ROOT, variants

sys.path.insert(0, str(ROOT / "tools"))
from check_dataset import audit_folder, find_leaks            # noqa: E402
from imagesorter.classifiers.faces import load_detector       # noqa: E402


def test_audit_flags_bad_photos_and_leaks(tmp_path, astronaut, cat):
    train, val = tmp_path / "train", tmp_path / "val"
    (train / "Eileen").mkdir(parents=True)
    (val / "Eileen").mkdir(parents=True)
    (train / "Nobody").mkdir()
    for i, v in enumerate(variants(astronaut, 3)):
        cv2.imwrite(str(train / "Eileen" / f"{i}.png"), v)
    cv2.imwrite(str(train / "Nobody" / "cat.png"), cat)
    (train / "Eileen" / "bad.jpg").write_bytes(b"junk")
    shutil.copy(train / "Eileen" / "0.png", val / "Eileen" / "same.png")

    det = load_detector()
    import pandas as pd
    df = pd.DataFrame(audit_folder(train, "train", det) + audit_folder(val, "val", det))
    status = dict(zip(df.file, df.status))
    assert status["cat.png"] == "no_face"
    assert status["bad.jpg"] == "unreadable"
    assert status["0.png"] == "ok"
    assert set(find_leaks(df).file) == {"0.png", "same.png"}
