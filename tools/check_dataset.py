"""Audit your photos BEFORE training: is each photo usable, are there enough,
and did the same picture end up in both train and val (which fakes good accuracy)?

    python tools/check_dataset.py
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                                      # noqa: E402

from imagesorter.classifiers.faces import (                              # noqa: E402
    default_cascade_path, detect_faces, load_detector, main_face, to_gray)
from imagesorter.imgio import read_image                                 # noqa: E402
from imagesorter.pipeline import IMAGE_EXTS                              # noqa: E402

TRAIN = ROOT / "Faces" / "train"
VAL = ROOT / "Faces" / "val"
MIN_FACE = 40
MIN_USABLE = 20      # photos per person worth training on


def audit_folder(folder, split, detector, min_face=MIN_FACE):
    """One row per image: ok / no_face / multiple_faces / unreadable."""
    rows = []
    folder = Path(folder)
    if not folder.is_dir():
        return rows
    for person in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in sorted(person.rglob("*")):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            img = read_image(f)
            if img is None:
                status = "unreadable"
            else:
                boxes = detect_faces(detector, to_gray(img), min_face)
                box, _ = main_face(boxes)          # same rule training uses
                status = "ok" if box else "no_face" if not boxes else "multiple_faces"
            rows.append({"split": split, "person": person.name, "file": f.name,
                         "status": status,
                         "md5": hashlib.md5(f.read_bytes()).hexdigest()})
    return rows


def find_leaks(df):
    """Same picture (identical bytes) present in train AND val."""
    both = df.groupby("md5")["split"].nunique()
    return df[df["md5"].isin(both[both > 1].index)]


def main():
    detector = load_detector(default_cascade_path())
    df = pd.DataFrame(audit_folder(TRAIN, "train", detector)
                      + audit_folder(VAL, "val", detector))
    if df.empty:
        print(f"No photos found in {TRAIN} or {VAL}")
        return

    table = (df.pivot_table(index=["split", "person"], columns="status",
                            values="file", aggfunc="count", fill_value=0)
               .reindex(columns=["ok", "no_face", "multiple_faces", "unreadable"],
                        fill_value=0))
    table.insert(0, "photos", table.sum(axis=1))
    print(table.to_string(), "\n")

    train_ok = table.loc["train"]["ok"] if "train" in table.index.get_level_values(0) else pd.Series(dtype=int)
    for person, ok in train_ok.items():
        if ok < MIN_USABLE:
            print(f"WARNING {person}: only {ok} usable training photos "
                  f"(aim for {MIN_USABLE}+)")
    train_people = set(df[df.split == "train"].person.str.lower().str.replace(" ", "_"))
    for person in sorted(set(df[df.split == "val"].person)):
        if person.lower().replace(" ", "_") not in train_people:
            print(f"NOTE    {person}: in val but not in train, so it counts as a stranger")

    leaks = find_leaks(df)
    if len(leaks):
        print(f"\nWARNING {leaks.md5.nunique()} picture(s) appear in BOTH train and val "
              f"(accuracy will look better than it is):")
        print(leaks[["split", "person", "file"]].to_string(index=False))
    else:
        print("\nNo train/val overlap found.")
    bad = df[df.status != "ok"]
    print(f"{len(bad)} of {len(df)} photos are not usable for training "
          f"(no face, several faces, or unreadable).")


if __name__ == "__main__":
    main()
