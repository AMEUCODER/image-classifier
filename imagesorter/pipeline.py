"""Pipeline: scan a folder, run a classifier on every image, copy each file
into output/<classifier>/<label>/, and return everything as a pandas DataFrame.

The pipeline knows nothing about faces: any object with a `name` and a
`classify(image) -> list[Result]` method can be plugged in.
"""
import filecmp
import re
import shutil
from pathlib import Path

import pandas as pd

from .imgio import read_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ERRORS_DIR = "_errors"
LOG_NAME = "sort_log.csv"
COLUMNS = ["source", "classifier", "label", "confidence", "distance",
           "destination", "status"]


def safe_folder_name(label):
    """Lowercase, spaces to underscores, drop anything a filesystem dislikes."""
    name = re.sub(r"[^\w\-]+", "_", str(label).strip().lower()).strip("_")
    return name or "unknown"


def scan_images(input_dir, skip_dir=None):
    """All image files under input_dir (recursive), except anything inside skip_dir."""
    input_dir = Path(input_dir)
    skip = Path(skip_dir).resolve() if skip_dir else None
    files = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if skip and skip in p.resolve().parents:
            continue
        files.append(p)
    return files


def unique_path(path):
    """photo.jpg -> photo_1.jpg -> photo_2.jpg ... if the name is taken."""
    if not path.exists():
        return path
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _place(src, dest_dir, dry_run, move):
    """Copy (or move) src into dest_dir; returns the final destination path.

    Name taken by a DIFFERENT file -> photo_1.jpg, photo_2.jpg ...
    Name taken by an IDENTICAL file (a re-run) -> reuse it, create nothing."""
    dest, n = dest_dir / src.name, 0
    while dest.exists():
        if filecmp.cmp(src, dest, shallow=False):
            if move and not dry_run:
                src.unlink(missing_ok=True)
            return dest
        n += 1
        dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        (shutil.move if move else shutil.copy2)(str(src), str(dest))
    return dest


def sort_images(input_dir, output_dir, classifier, dry_run=False, move=False,
                min_confidence=0.0, progress=None):
    """Classify and sort every image. Returns a DataFrame (one row per image/label).

    dry_run         : only report where files WOULD go
    move            : move instead of copy (default is copy: originals stay safe)
    min_confidence  : results below this go to 'uncertain' instead of their label
    progress        : optional callback(done, total, path) for a UI
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    files = scan_images(input_dir, skip_dir=output_dir)
    root = output_dir / classifier.name
    rows = []

    for i, src in enumerate(files, 1):
        image = read_image(src)
        if image is None:
            dest = _place(src, output_dir / ERRORS_DIR, dry_run, move)
            rows.append([str(src), classifier.name, ERRORS_DIR, 0.0, None,
                         str(dest), "unreadable"])
        else:
            for r in classifier.classify(image):
                label = safe_folder_name(r.label)
                status = "ok"
                if r.confidence < min_confidence:
                    label, status = "uncertain", "low_confidence"
                dest = _place(src, root / label, dry_run, move=False)
                rows.append([str(src), classifier.name, label,
                             round(r.confidence, 3),
                             r.details.get("distance"), str(dest), status])
            if move and not dry_run:
                src.unlink(missing_ok=True)   # moved only after all copies exist
        if progress:
            progress(i, len(files), src)

    df = pd.DataFrame(rows, columns=COLUMNS)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / LOG_NAME, index=False)
    return df


def summary(df):
    """Small table: images per folder."""
    if df.empty:
        return "No images found."
    counts = df.groupby("label").size().sort_values(ascending=False)
    lines = [f"{'folder':<24}{'images':>7}", "-" * 31]
    lines += [f"{label:<24}{n:>7}" for label, n in counts.items()]
    lines.append("-" * 31)
    lines.append(f"{'files processed':<24}{df['source'].nunique():>7}")
    return "\n".join(lines)
