"""Tests for the face classifier. Real detection tests use the shipped Haar
cascade and a real face photo; logic tests use a fake detector."""
import cv2
import numpy as np
import pytest

from imagesorter.classifiers.faces import (
    FaceClassifier, detect_faces, load_detector, load_labels, to_gray)


class WholeImageDetector:
    """Pretends the entire image is one face (for synthetic data)."""
    def detectMultiScale(self, gray, *a, **k):
        return np.array([[0, 0, gray.shape[1], gray.shape[0]]])


class FixedBoxes:
    def __init__(self, boxes):
        self.boxes = np.array(boxes).reshape(-1, 4)

    def detectMultiScale(self, *a, **k):
        return self.boxes


def synthetic_person(folder, seed, n=6):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed + 1000)
    base = np.random.default_rng(seed).integers(0, 255, (110, 110), dtype=np.uint8)
    for i in range(n):
        noisy = np.clip(base.astype(int) + rng.integers(-12, 12, base.shape), 0, 255)
        cv2.imwrite(str(folder / f"{i}.png"), noisy.astype(np.uint8))
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


# ---------------- real detection ----------------
def test_haar_cascade_finds_one_real_face(astronaut):
    boxes = detect_faces(load_detector(), to_gray(astronaut), 40)
    assert len(boxes) == 1


def test_train_then_recognize_real_face(tmp_path, eileen_dataset, astronaut, cat):
    model, labels = tmp_path / "m.yml", tmp_path / "l.json"
    report = FaceClassifier.train(eileen_dataset, model, labels)
    assert report["counts"]["Eileen Collins"] > 0

    clf = FaceClassifier(model, labels, threshold=120)
    assert [r.label for r in clf.classify(astronaut)] == ["eileen_collins"]
    assert [r.label for r in clf.classify(cat)] == ["no_face"]


# ---------------- logic with synthetic faces ----------------
def test_train_add_person_and_unknown(tmp_path):
    data = tmp_path / "data"
    a = synthetic_person(data / "ann_lee", 1)
    b = synthetic_person(data / "bob", 2)
    model, labels = tmp_path / "m.yml", tmp_path / "l.json"

    FaceClassifier.train(data, model, labels, detect=False)
    clf = FaceClassifier(model, labels, detector=WholeImageDetector())
    assert clf.classify(a)[0].label == "ann_lee"
    assert clf.classify(b)[0].label == "bob"
    stranger = cv2.cvtColor(
        np.random.default_rng(99).integers(0, 255, (110, 110), dtype=np.uint8),
        cv2.COLOR_GRAY2BGR)
    assert clf.classify(stranger)[0].label == "unknown_person"

    c = synthetic_person(tmp_path / "cara_photos", 3)
    FaceClassifier.add_person(tmp_path / "cara_photos", "cara", model, labels,
                              detect=False)
    clf = FaceClassifier(model, labels, detector=WholeImageDetector())
    assert clf.classify(c)[0].label == "cara"
    assert clf.classify(a)[0].label == "ann_lee"      # old people still known
    assert load_labels(labels)[0] == {0: "ann_lee", 1: "bob", 2: "cara"}


@pytest.mark.parametrize("policy,expected", [
    ("group", ["group"]),
    ("largest", ["ann_lee"]),
])
def test_multi_face_policies(tmp_path, policy, expected):
    data = tmp_path / "data"
    a = synthetic_person(data / "ann_lee", 1)
    synthetic_person(data / "bob", 2)
    model, labels = tmp_path / "m.yml", tmp_path / "l.json"
    FaceClassifier.train(data, model, labels, detect=False)

    clf = FaceClassifier(model, labels, policy=policy,
                         detector=FixedBoxes([(0, 0, 110, 110), (10, 10, 60, 60)]))
    assert [r.label for r in clf.classify(a)] == expected


def test_bad_policy_rejected(tmp_path):
    data = tmp_path / "data"
    synthetic_person(data / "ann", 1)
    FaceClassifier.train(data, tmp_path / "m.yml", tmp_path / "l.json", detect=False)
    with pytest.raises(ValueError):
        FaceClassifier(tmp_path / "m.yml", tmp_path / "l.json", policy="nope",
                       detector=WholeImageDetector())


def test_sync_trains_only_what_is_missing(tmp_path):
    from imagesorter.classifiers.faces import sync_dataset
    data = tmp_path / "train"
    a = synthetic_person(data / "Ann Lee", 1)
    synthetic_person(data / "bob", 2)
    model, labels = tmp_path / "m.yml", tmp_path / "l.json"
    det = WholeImageDetector()

    first = sync_dataset(data, model, labels, detector=det)
    assert set(first["counts"]) == {"Ann Lee", "bob"}            # from scratch

    assert sync_dataset(data, model, labels, detector=det)["counts"] == {}  # no-op

    c = synthetic_person(data / "Cara_Jones", 3)                 # new folder appears
    third = sync_dataset(data, model, labels, detector=det)
    assert set(third["counts"]) == {"Cara_Jones"}                # only the new one

    clf = FaceClassifier(model, labels, detector=det)
    assert clf.classify(c)[0].label == "cara_jones"
    assert clf.classify(a)[0].label == "ann_lee"


def test_add_person_matches_existing_name_loosely(tmp_path):
    data = tmp_path / "data"
    synthetic_person(data / "Ann Lee", 1)
    model, labels = tmp_path / "m.yml", tmp_path / "l.json"
    FaceClassifier.train(data, model, labels, detect=False)
    more = tmp_path / "more"
    synthetic_person(more, 1)
    FaceClassifier.add_person(more, "ann_lee", model, labels, detect=False)
    assert load_labels(labels)[0] == {0: "Ann Lee"}             # no duplicate person


def test_read_image_handles_unicode_path(tmp_path):
    from imagesorter.imgio import read_image
    p = tmp_path / "صورة_é.png"
    cv2.imwrite(str(tmp_path / "t.png"), np.full((20, 20, 3), 90, np.uint8))
    (tmp_path / "t.png").rename(p)
    assert read_image(p).shape == (20, 20, 3)
    assert read_image(tmp_path / "missing.png") is None
    (tmp_path / "empty.png").write_bytes(b"")
    assert read_image(tmp_path / "empty.png") is None


# ---------------- regressions found with real use ----------------
def test_faces_on_dark_and_bright_backgrounds_are_found_and_recognized(
        tmp_path, eileen_dataset, astronaut):
    """v1 equalized the whole photo before detecting and missed these entirely."""
    from conftest import studio_backgrounds
    model, labels = tmp_path / "m.yml", tmp_path / "l.json"
    FaceClassifier.train(eileen_dataset, model, labels)
    clf = FaceClassifier(model, labels, threshold=120)
    for name, img in studio_backgrounds(astronaut).items():
        assert [r.label for r in clf.classify(img)] == ["eileen_collins"], name


def test_same_photo_gives_same_answer_in_training_and_sorting(
        tmp_path, eileen_dataset, astronaut):
    """A photo that trained the model must never come back as no_face."""
    model, labels = tmp_path / "m.yml", tmp_path / "l.json"
    FaceClassifier.train(eileen_dataset, model, labels)
    clf = FaceClassifier(model, labels, threshold=120)
    for f in sorted((eileen_dataset / "Eileen Collins").iterdir()):
        img = cv2.imread(str(f))
        assert clf.classify(img)[0].label == "eileen_collins", f.name


def test_main_face_rule():
    from imagesorter.classifiers.faces import main_face
    assert main_face([])[0] is None
    assert main_face([(0, 0, 90, 90)])[0] == (0, 0, 90, 90)
    big, small = (0, 0, 100, 100), (5, 5, 40, 40)
    assert main_face([big, small])[0] == big                 # clearly biggest: use it
    assert main_face([big, (0, 0, 95, 95)])[0] is None       # two similar: ambiguous


def test_report_counts_photos_used(tmp_path):
    data = tmp_path / "data"
    synthetic_person(data / "ann", 1, n=4)
    (data / "ann" / "broken.jpg").write_bytes(b"junk")
    report = FaceClassifier.train(data, tmp_path / "m.yml", tmp_path / "l.json",
                                  detect=False)
    assert report["photos"]["ann"] == (4, 5)
    assert report["skipped"]["ann"] == [("broken.jpg", "unreadable")]


def test_old_model_is_retrained_automatically(tmp_path):
    import json
    from imagesorter.classifiers.faces import sync_dataset
    data = tmp_path / "data"
    synthetic_person(data / "ann", 1)
    model, labels = tmp_path / "m.yml", tmp_path / "l.json"
    FaceClassifier.train(data, model, labels, detect=False)

    meta = json.loads(labels.read_text())
    meta["prep"] = 1                                        # pretend: made by v1
    labels.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="older version"):
        FaceClassifier(model, labels, detector=WholeImageDetector())

    report = sync_dataset(data, model, labels, detect=False)
    assert report["notes"] and (tmp_path / "m.yml.old").exists()
    FaceClassifier(model, labels, detector=WholeImageDetector())   # loads fine now
