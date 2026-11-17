"""Recognize faces with LBPH, and auto-train anyone new in the `people` list.

HOW TO ADD A PERSON
  1. Add the name at the END of `people` (never in the middle, see note below).
  2. Put their photos in  ../Faces/train/<their name>/   (case / spaces / _ don't matter).
  3. Run this script. It trains only the new person, saves the model, then recognizes.

NOTE: a person's position in the list IS their ID inside face_trained.yml.
      Inserting in the middle or reordering would make everyone's names wrong.
"""
import shutil
from pathlib import Path

import numpy as np
import cv2 as cv

# ---------------- paths (relative to this script, not to where you run it) ----
HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / 'models'
HAAR_PATH = MODELS / 'haar_face.xml'
MODEL_PATH = MODELS / 'face_trained.yml'
TRAIN_DIR = HERE.parent / 'Faces' / 'train'
TEST_IMAGE = HERE.parent / 'Faces' / 'val' / 'madonna' / '4.jpg'

# ---------------- settings ----------------------------------------------------
# Add new people at the END only. Position = ID stored in the model.
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# LBPH "confidence" is really a DISTANCE: lower = better match.
# None = always answer with the closest person. Set a number (try 80-110 after
# looking at the printed values) to answer "Unknown" for strangers.
THRESHOLD = None

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

haar_cascade = None   # loaded in main()


# ---------------- helpers -----------------------------------------------------
def norm(name):
    """'Ben Afflek', 'ben_afflek' and 'BEN AFFLEK' all become 'benafflek'."""
    return name.lower().replace(' ', '').replace('_', '')


def find_person_folder(name):
    if not TRAIN_DIR.is_dir():
        return None
    for d in TRAIN_DIR.iterdir():
        if d.is_dir() and norm(d.name) == norm(name):
            return d
    return None


def faces_from_folder(folder):
    """Gray face crops, same preprocessing as recognition (no resize, no equalize).
    Uses the largest face per photo so stray false detections don't pollute training."""
    crops = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        img = cv.imread(str(f))
        if img is None:
            print(f'  skipped (unreadable): {f.name}')
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rects = haar_cascade.detectMultiScale(gray, 1.1, 4)
        if len(rects) == 0:
            print(f'  skipped (no face): {f.name}')
            continue
        x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
        crops.append(gray[y:y + h, x:x + w])
    return crops


def known_ids(recognizer):
    """IDs already learned by the loaded model."""
    labels = recognizer.getLabels()
    return set(int(v) for v in np.array(labels).flatten())


def sync_model(recognizer, model_exists):
    """Train every person in `people` the model doesn't know yet.
    Returns True if the model changed and must be saved."""
    known = known_ids(recognizer) if model_exists else set()
    has_data = model_exists and bool(known)
    changed = False

    for idx, name in enumerate(people):
        if idx in known:
            continue
        folder = find_person_folder(name)
        if folder is None:
            print(f'[new] {name}: no folder found in {TRAIN_DIR} - skipped')
            continue
        print(f'[new] training {name} (id {idx}) from {folder.name}/')
        crops = faces_from_folder(folder)
        if not crops:
            print(f'[new] {name}: no usable faces - skipped')
            continue
        ids = np.array([idx] * len(crops))
        if has_data:
            recognizer.update(crops, ids)      # adds to the existing model
        else:
            recognizer.train(crops, ids)       # very first person(s)
            has_data = True
        changed = True
        print(f'[new] {name}: learned from {len(crops)} photos')
    return changed


def save_model(recognizer):
    if MODEL_PATH.exists():
        shutil.copy(MODEL_PATH, MODEL_PATH.with_suffix('.yml.bak'))   # safety copy
    recognizer.write(str(MODEL_PATH))
    print(f'Model saved: {MODEL_PATH}')


def recognize(recognizer, img):
    """Returns the annotated image and prints every face found."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for (x, y, w, h) in haar_cascade.detectMultiScale(gray, 1.1, 4):
        label, distance = recognizer.predict(gray[y:y + h, x:x + w])
        if (THRESHOLD is not None and distance > THRESHOLD) or label >= len(people):
            who = 'Unknown'
        else:
            who = people[label]
        print(f'Label = {who} with a distance of {distance:.1f} (lower = better)')
        cv.putText(img, who, (x, max(y - 10, 20)), cv.FONT_HERSHEY_COMPLEX,
                   1.0, (0, 255, 0), thickness=2)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    return img


# ---------------- main --------------------------------------------------------
def main():
    global haar_cascade
    haar_cascade = cv.CascadeClassifier(str(HAAR_PATH))
    if haar_cascade.empty():
        raise SystemExit(f'Could not load Haar cascade: {HAAR_PATH}')

    recognizer = cv.face.LBPHFaceRecognizer_create()
    model_exists = MODEL_PATH.exists()
    if model_exists:
        recognizer.read(str(MODEL_PATH))

    if sync_model(recognizer, model_exists):
        save_model(recognizer)
    elif not model_exists:
        raise SystemExit('No model yet and no training photos found.')

    img = cv.imread(str(TEST_IMAGE))
    if img is None:
        raise SystemExit(f'Could not read test image: {TEST_IMAGE}')
    cv.imshow('Detected Face', recognize(recognizer, img))
    cv.waitKey(0)


if __name__ == '__main__':
    main()
