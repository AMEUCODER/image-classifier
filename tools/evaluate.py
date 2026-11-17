"""How good is the model? Every photo in Faces/val/<person>/ is classified and
compared with its folder name. Prints accuracy, a per-person table and a
threshold sweep, and saves output/evaluation.csv.

    python tools/evaluate.py

A val folder for someone the model does not know counts as a stranger: the
right answer for those photos is 'unknown_person'.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

import main as settings                                                  # noqa: E402
from imagesorter.classifiers.faces import (                              # noqa: E402
    NO_FACE, UNKNOWN, FaceClassifier, _norm, clean_name)
from imagesorter.imgio import read_image                                 # noqa: E402
from imagesorter.pipeline import IMAGE_EXTS                              # noqa: E402

SWEEP = list(range(40, 151, 10))


def collect(val_dir, classifier):
    """Nearest person + distance for every val photo (threshold disabled)."""
    known = {_norm(n) for n in classifier.labels.values()}
    rows = []
    for person in sorted(p for p in Path(val_dir).iterdir() if p.is_dir()):
        for f in sorted(person.rglob("*")):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            img = read_image(f)
            if img is None:
                continue
            r = classifier.classify(img)[0]
            truth_known = _norm(person.name) in known
            rows.append({"person": person.name, "file": f.name,
                         "truth": clean_name(person.name) if truth_known else UNKNOWN,
                         "nearest": r.label, "no_face": r.label == NO_FACE,
                         "distance": r.details.get("distance", np.nan)})
    return pd.DataFrame(rows)


def predict_at(df, threshold):
    """Final label for each photo if the threshold were `threshold`."""
    far = df["distance"] > threshold
    return df["nearest"].where(~far, UNKNOWN).where(~df["no_face"], NO_FACE)


def sweep(df, thresholds=SWEEP):
    rows = []
    faces = df[~df.no_face]
    n = max(len(faces), 1)
    for t in thresholds:
        pred = predict_at(faces, t)
        known = faces.truth != UNKNOWN
        right_known = ((pred == faces.truth) & known).sum()
        wrong_known = ((pred != faces.truth) & known & (pred != UNKNOWN)).sum()
        missed_known = ((pred == UNKNOWN) & known).sum()
        strangers = (~known).sum()
        let_in = ((pred != UNKNOWN) & ~known).sum()
        rows.append({"threshold": t, "correct": right_known, "wrong_name": wrong_known,
                     "called_unknown": missed_known, "strangers": strangers,
                     "stranger_let_in": let_in,
                     "score": (right_known - wrong_known - let_in) / n})
    return pd.DataFrame(rows)


def main():
    model, labels = settings.MODEL, settings.LABELS
    if not (model.is_file() and labels.is_file()):
        sys.exit("No trained model yet: run  python main.py  first.")
    if not settings.INPUT_FOLDER.is_dir():
        sys.exit(f"Val folder not found: {settings.INPUT_FOLDER}")

    clf = FaceClassifier(model, labels, threshold=1e9, policy="largest",
                         min_face=settings.MIN_FACE, cascade_path=settings.CASCADE)
    df = collect(settings.INPUT_FOLDER, clf)
    if df.empty:
        sys.exit("No usable photos found in the val folder.")

    T = settings.THRESHOLD
    df["predicted"] = predict_at(df, T)
    faces = df[~df.no_face]
    acc = (faces.predicted == faces.truth).mean() if len(faces) else float("nan")

    print(f"Photos: {len(df)}   no face found: {int(df.no_face.sum())}   "
          f"threshold in use: {T}")
    print(f"Accuracy on photos with a face: {acc:.1%}\n")
    per = (faces.assign(correct=faces.predicted == faces.truth)
                .groupby("person")["correct"].agg(photos="count", correct="sum"))
    per["accuracy"] = (per.correct / per.photos).map("{:.0%}".format)
    print(per.to_string(), "\n")

    sw = sweep(df)
    print("Threshold sweep (score = correct - wrong names - strangers let in, per photo):")
    print(sw.to_string(index=False))
    if sw.strangers.iloc[0] == 0:
        print("\nNo stranger photos in Faces/val, so no threshold recommendation. "
              "Add a val folder of a person who is NOT in Faces/train to get one.")
    else:
        tied = sw[sw.score == sw.score.max()].threshold
        best = int(tied.iloc[len(tied) // 2])          # middle of the best range
        print(f"\nBest threshold on this data: about {best} "
              f"(range {tied.min()} to {tied.max()}). Set THRESHOLD = {best} in main.py")
    if len(faces) < 30:
        print("Note: very few photos, so treat this as a rough guide only.")

    out = settings.OUTPUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "evaluation.csv", index=False)
    print(f"Per-photo results: {out / 'evaluation.csv'}")


if __name__ == "__main__":
    main()
