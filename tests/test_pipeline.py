import cv2

from imagesorter.classifiers.faces import FaceClassifier
from imagesorter.pipeline import safe_folder_name, scan_images, sort_images, unique_path


def test_safe_folder_name():
    assert safe_folder_name("Ben Afflek!") == "ben_afflek"
    assert safe_folder_name("   ") == "unknown"


def test_unique_path(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert unique_path(tmp_path / "a.jpg").name == "a_1.jpg"


def test_sort_end_to_end(tmp_path, eileen_dataset, astronaut, cat):
    model, labels = tmp_path / "m.yml", tmp_path / "l.json"
    FaceClassifier.train(eileen_dataset, model, labels)
    clf = FaceClassifier(model, labels, threshold=120)

    inbox = tmp_path / "inbox"
    (inbox / "sub").mkdir(parents=True)
    cv2.imwrite(str(inbox / "face.png"), astronaut)
    cv2.imwrite(str(inbox / "sub" / "face.png"),               # same name, different picture
                cv2.convertScaleAbs(astronaut, alpha=1.0, beta=5))
    cv2.imwrite(str(inbox / "cat.png"), cat)
    (inbox / "broken.jpg").write_bytes(b"not an image")
    (inbox / "notes.txt").write_text("ignored")

    out = tmp_path / "out"
    assert len(scan_images(inbox)) == 4

    dry = sort_images(inbox, out, clf, dry_run=True)
    assert not out.exists() and len(dry) == 4

    df = sort_images(inbox, out, clf)
    assert len(list((out / "faces" / "eileen_collins").iterdir())) == 2
    assert len(list((out / "faces" / "no_face").iterdir())) == 1
    assert len(list((out / "_errors").iterdir())) == 1
    assert (out / "sort_log.csv").is_file()
    assert (inbox / "face.png").exists()                  # originals untouched
    assert set(df["status"]) == {"ok", "unreadable"}
    assert len(list((out / "faces" / "eileen_collins").iterdir())) == 2

    sort_images(inbox, out, clf)                          # re-run: no duplicates
    assert len(list((out / "faces" / "eileen_collins").iterdir())) == 2
    assert len(list((out / "faces" / "no_face").iterdir())) == 1
