"""Face classifier: Haar cascade detection + LBPH recognition, with training.

Needs opencv-contrib-python (cv2.face lives in the contrib build).

Dataset layout for training (folder name = person name = output folder name):

    dataset/
      ben_affleck/   img1.jpg img2.jpg ...
      madonna/       ...

Command line:

    python -m imagesorter.classifiers.faces train dataset models/
    python -m imagesorter.classifiers.faces add   new_person_dir models/ --name jane_doe
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..imgio import read_image
from .base import BaseClassifier, Result

UNKNOWN = "unknown_person"
NO_FACE = "no_face"
GROUP = "group"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 1200          # bigger photos are shrunk before detection (much faster)
DEFAULT_MODEL = "faces_trained.yml"
DEFAULT_LABELS = "faces_labels.json"

# Bump when the way faces are prepared for the recognizer changes: models made
# with another version are retrained automatically (old files kept as *.old).
PREP_VERSION = 2

# Detection tries these in order and stops at the first that finds a face:
# (scaleFactor, minNeighbors, accept only when exactly one face is found)
# Pass 1 is exactly what the original course script did: it found a face in 10 of 11
# test pictures with no false faces in non-face pictures. Pass 2 is more forgiving
# (small or awkward faces) but is only trusted when it finds ONE face, because on
# non-face pictures it tends to produce several false ones.
# Measured and rejected: equalizing the whole photo before detection (fewer faces
# found AND more false faces).
DETECT_PASSES = [(1.1, 4, False), (1.05, 3, True)]
BIGGER_FACTOR = 1.5      # in training, the main face must be this much bigger than the next


# ----------------------------------------------------------------------
# helpers shared by training and classification (same preprocessing = same
# numbers, which is what makes recognition reliable)
# ----------------------------------------------------------------------
def default_cascade_path():
    """Explicit path wins; then the project's models/haar_face.xml (shipped in
    the zip); then the copy bundled with OpenCV (missing in some builds)."""
    project = Path(__file__).resolve().parents[2] / "models" / "haar_face.xml"
    if project.is_file():
        return project
    return Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def load_detector(cascade_path=None):
    cascade = cascade_path or default_cascade_path()
    detector = cv2.CascadeClassifier(str(cascade))
    if detector.empty():
        raise RuntimeError(f"Could not load cascade: {cascade}")
    return detector


def to_gray(image, max_side=MAX_SIDE):
    """Plain grayscale (NOT equalized), shrinking huge photos first."""
    longest = max(image.shape[:2])
    if max_side and longest > max_side:
        scale = max_side / longest
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def detect_faces(detector, gray, min_face, return_pass=False):
    """Faces as (x, y, w, h), biggest first. Tries DETECT_PASSES until one finds
    something. With return_pass=True also returns the index of the winning pass."""
    for i, (scale, neighbors, single_only) in enumerate(DETECT_PASSES):
        boxes = detector.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors,
            minSize=(min_face, min_face))
        if single_only and len(boxes) != 1:
            continue
        if len(boxes):
            found = sorted((tuple(int(v) for v in b) for b in boxes),
                           key=lambda b: b[2] * b[3], reverse=True)
            return (found, i) if return_pass else found
    return ([], None) if return_pass else []


def main_face(boxes):
    """The face to learn from in a TRAINING photo: (box, None) or (None, reason).
    Several faces are fine when one is clearly the biggest."""
    if not boxes:
        return None, "no face found"
    if len(boxes) == 1:
        return boxes[0], None
    area = lambda b: b[2] * b[3]
    if area(boxes[0]) >= BIGGER_FACTOR * area(boxes[1]):
        return boxes[0], None
    return None, f"{len(boxes)} faces of similar size (ambiguous)"


def face_patch(gray, box, face_size):
    """Crop one face, resize, and equalize THE FACE ONLY. Normalizing the face
    (not the whole photo) keeps a dark or bright background from changing it."""
    x, y, w, h = box
    face = gray[y:y + h, x:x + w]
    if face_size:
        face = cv2.resize(face, face_size)
    return cv2.equalizeHist(face)


def _norm(name):
    """'Ben Afflek', 'ben_afflek' and 'BEN AFFLEK' are the same person."""
    return name.lower().replace(" ", "").replace("_", "")


def clean_name(name):
    return name.strip().lower().replace(" ", "_")


def save_labels(path, labels, face_size):
    data = {"prep": PREP_VERSION,
            "face_size": list(face_size) if face_size else None,
            "labels": {str(k): v for k, v in sorted(labels.items())}}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_meta(path):
    """{"labels", "face_size", "prep", "own_format"}. Reads our own json format and
    also old plain list / dict / txt files (then own_format is False)."""
    path = Path(path)
    meta = {"labels": {}, "face_size": None, "prep": 0, "own_format": False}
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "labels" in data:
            meta.update(labels={int(k): v for k, v in data["labels"].items()},
                        face_size=tuple(data["face_size"]) if data.get("face_size") else None,
                        prep=int(data.get("prep", 0)), own_format=True)
        elif isinstance(data, dict):
            meta["labels"] = {int(k): v for k, v in data.items()}
        else:
            meta["labels"] = dict(enumerate(data))
    else:
        names = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        meta["labels"] = dict(enumerate(names))
    return meta


def load_labels(path):
    """Returns (labels dict, face_size or None)."""
    meta = load_meta(path)
    return meta["labels"], meta["face_size"]


def collect_faces(folder, detector, face_size, min_face=40,
                  detect=True, augment=True):
    """Read every image in a folder. Returns (crops, skipped, total_photos).

    detect=True   -> use the main face in each photo (see main_face)
    detect=False  -> treat each image as an already-cropped face
    augment=True  -> also add a mirrored copy of each face (more data)
    """
    crops, skipped, total = [], [], 0
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        total += 1
        img = read_image(f)
        if img is None:
            skipped.append((f.name, "unreadable"))
            continue
        gray = to_gray(img)
        if detect:
            box, why = main_face(detect_faces(detector, gray, min_face))
            if box is None:
                skipped.append((f.name, why))
                continue
        else:
            box = (0, 0, gray.shape[1], gray.shape[0])
        face = face_patch(gray, box, face_size)
        crops.append(face)
        if augment:
            crops.append(cv2.flip(face, 1))
    return crops, skipped, total


# ----------------------------------------------------------------------
# the classifier
# ----------------------------------------------------------------------
class FaceClassifier(BaseClassifier):
    name = "faces"

    def __init__(self, model_path, labels_path, threshold=80.0,
                 policy="all", face_size=None, min_face=40,
                 cascade_path=None, detector=None):
        """
        model_path : trained LBPH file (.yml)
        labels_path: labels .json written by train() (old list/txt also works)
        threshold  : LBPH distance limit; LOWER distance = better match
        policy     : 'all'     -> one label per recognized person
                     'largest' -> only the biggest face counts
                     'group'   -> more than one face -> single 'group' label
        face_size  : override crop size; default = the size stored at training
        detector   : custom detector object (anything with detectMultiScale)
        """
        if policy not in ("all", "largest", "group"):
            raise ValueError("policy must be 'all', 'largest' or 'group'")
        self.threshold = threshold
        self.policy = policy
        self.min_face = min_face

        meta = load_meta(labels_path)
        if meta["own_format"] and meta["prep"] != PREP_VERSION:
            raise RuntimeError(
                "This model was trained with an older version of the face "
                "preprocessing. Delete models/faces_trained.yml and "
                "models/faces_labels.json and train again (main.py does it for you).")
        self.labels = meta["labels"]
        self.face_size = tuple(face_size) if face_size else meta["face_size"]

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(str(model_path))
        self.detector = detector or load_detector(cascade_path)

    # ---------- classify ----------
    def classify(self, image):
        gray = to_gray(image)
        boxes = detect_faces(self.detector, gray, self.min_face)

        if not boxes:
            return [Result(NO_FACE, 1.0)]
        if self.policy == "group" and len(boxes) > 1:
            return [Result(GROUP, 1.0, {"faces": len(boxes)})]
        if self.policy == "largest":
            boxes = boxes[:1]

        results = [self._recognize(gray, b) for b in boxes]

        seen, unique = set(), []          # one Result per distinct label
        for r in results:
            if r.label not in seen:
                seen.add(r.label)
                unique.append(r)
        return unique

    def _recognize(self, gray, box):
        face = face_patch(gray, box, self.face_size)
        label_id, distance = self.recognizer.predict(face)
        details = {"distance": float(distance), "box": box,
                   "nearest": clean_name(self.labels.get(label_id, "?"))}

        if distance <= self.threshold and label_id in self.labels:
            confidence = max(0.0, 1.0 - distance / 100.0)
            return Result(clean_name(self.labels[label_id]), confidence, details)
        return Result(UNKNOWN, 0.0, details)

    # ---------- training ----------
    @staticmethod
    def train(dataset_dir, model_path, labels_path, face_size=(100, 100),
              min_face=40, detect=True, augment=True, cascade_path=None,
              detector=None):
        """Train a NEW model from scratch. One sub-folder per person.
        Returns a report: counts (face crops per person), photos (used, total),
        skipped (file, reason) and notes."""
        detector = detector or (load_detector(cascade_path) if detect else None)

        faces, ids, labels = [], [], {}
        report = {"counts": {}, "photos": {}, "skipped": {}, "notes": []}
        people = sorted(p for p in Path(dataset_dir).iterdir() if p.is_dir())
        for person in people:
            crops, skipped, total = collect_faces(person, detector, face_size,
                                                  min_face, detect, augment)
            report["skipped"][person.name] = skipped
            report["photos"][person.name] = (total - len(skipped), total)
            if not crops:
                continue                      # nobody to learn for this folder
            pid = len(labels)
            labels[pid] = person.name
            faces += crops
            ids += [pid] * len(crops)
            report["counts"][person.name] = len(crops)

        if not faces:
            raise RuntimeError("No usable training faces found.")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        recognizer.write(str(model_path))
        save_labels(labels_path, labels, face_size)
        return report

    @staticmethod
    def add_person(images_dir, name, model_path, labels_path, min_face=40,
                   detect=True, augment=True, cascade_path=None, detector=None):
        """Teach an EXISTING model one more person without retraining the rest.
        If the name already exists, the new photos are added to that person."""
        labels, face_size = load_labels(labels_path)
        detector = detector or (load_detector(cascade_path) if detect else None)

        crops, skipped, total = collect_faces(images_dir, detector, face_size,
                                              min_face, detect, augment)
        if not crops:
            raise RuntimeError("No usable faces in the new folder.")

        by_name = {_norm(v): k for k, v in labels.items()}
        pid = by_name.get(_norm(name))
        if pid is None:
            pid = max(labels, default=-1) + 1
            labels[pid] = name

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(model_path))
        recognizer.update(crops, np.array([pid] * len(crops)))
        recognizer.write(str(model_path))
        save_labels(labels_path, labels, face_size)
        return {"counts": {name: len(crops)},
                "photos": {name: (total - len(skipped), total)},
                "skipped": {name: skipped}, "notes": []}


def sync_dataset(dataset_dir, model_path, labels_path, face_size=(100, 100),
                 min_face=40, detect=True, augment=True, cascade_path=None,
                 detector=None):
    """Make the model match the dataset folder, training only what is missing.

    - no model yet        -> train everyone from scratch
    - model exists        -> train only the person folders it does not know yet
    - nothing new         -> do nothing
    Returns the same report dict as train() (empty "counts" = already up to date).
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Training folder not found: {dataset_dir}")
    common = dict(min_face=min_face, detect=detect, augment=augment,
                  cascade_path=cascade_path, detector=detector)

    if not (Path(model_path).is_file() and Path(labels_path).is_file()):
        return FaceClassifier.train(dataset_dir, model_path, labels_path,
                                    face_size=face_size, **common)

    meta = load_meta(labels_path)
    if meta["own_format"] and meta["prep"] != PREP_VERSION:
        Path(model_path).replace(str(model_path) + ".old")
        Path(labels_path).replace(str(labels_path) + ".old")
        report = FaceClassifier.train(dataset_dir, model_path, labels_path,
                                      face_size=face_size, **common)
        report["notes"].append("Model was made by an older version, so it was "
                               "retrained from scratch (old files kept as *.old).")
        return report

    known = {_norm(n) for n in meta["labels"].values()}
    report = {"counts": {}, "photos": {}, "skipped": {}, "notes": []}
    for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if _norm(folder.name) in known:
            continue
        try:
            r = FaceClassifier.add_person(folder, folder.name, model_path,
                                          labels_path, **common)
        except RuntimeError:
            report["skipped"][folder.name] = [("(whole folder)", "no usable faces")]
            report["photos"][folder.name] = (0, 0)
            continue
        for key in ("counts", "photos", "skipped"):
            report[key].update(r[key])
    return report


# module-level names that classifiers/__init__.py exports
train_model = FaceClassifier.train
add_person = FaceClassifier.add_person


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def _print_report(report):
    for note in report.get("notes", []):
        print(f"  NOTE: {note}")
    for name, n in report["counts"].items():
        used, total = report["photos"][name]
        print(f"  {name}: {used} of {total} photos used ({n} face crops)")
    for name, items in report["skipped"].items():
        for fname, why in items:
            print(f"  skipped {name}/{fname}: {why}")


def _main():
    p = argparse.ArgumentParser(description="Train or extend the face model")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train from a dataset folder")
    t.add_argument("dataset")
    t.add_argument("models_dir")
    t.add_argument("--size", type=int, default=100)
    t.add_argument("--no-detect", action="store_true",
                   help="images are already cropped faces")

    a = sub.add_parser("add", help="add one person to an existing model")
    a.add_argument("images_dir")
    a.add_argument("models_dir")
    a.add_argument("--name", required=True)
    a.add_argument("--no-detect", action="store_true")

    args = p.parse_args()
    models = Path(args.models_dir)
    model, labels = models / DEFAULT_MODEL, models / DEFAULT_LABELS

    if args.cmd == "train":
        report = FaceClassifier.train(args.dataset, model, labels,
                                      face_size=(args.size, args.size),
                                      detect=not args.no_detect)
    else:
        report = FaceClassifier.add_person(args.images_dir, args.name, model,
                                           labels, detect=not args.no_detect)
    _print_report(report)
    print(f"Saved: {model} and {labels}")


if __name__ == "__main__":
    _main()
