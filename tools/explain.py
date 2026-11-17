"""See exactly what the program does with ONE photo: whether a face is found,
which detection pass found it, how close it is to each trained person, and why
it ended up as a name, unknown_person or no_face. Also saves an annotated copy.

    python tools/explain.py "C:\\path\\to\\photo.jpg"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2                                                               # noqa: E402

import main as settings                                                  # noqa: E402
from imagesorter.classifiers.faces import (                              # noqa: E402
    DETECT_PASSES, FaceClassifier, clean_name, detect_faces, face_patch, to_gray)
from imagesorter.imgio import read_image                                 # noqa: E402


def ranking(clf, patch):
    """Best (smallest) distance per person, closest first."""
    collector = cv2.face.StandardCollector_create()
    clf.recognizer.predict_collect(patch, collector)
    best = {}
    for label, dist in collector.getResults(True):
        best[label] = min(dist, best.get(label, 1e18))
    return sorted(((clf.labels.get(k, "?"), d) for k, d in best.items()), key=lambda t: t[1])


def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: python tools/explain.py "path\\to\\photo.jpg"')
    path = Path(sys.argv[1])
    img = read_image(path)
    if img is None:
        sys.exit(f"Cannot read this file as an image: {path}")
    if not (settings.MODEL.is_file() and settings.LABELS.is_file()):
        sys.exit("No trained model yet: run  python main.py  first.")

    clf = FaceClassifier(settings.MODEL, settings.LABELS, threshold=settings.THRESHOLD,
                         policy="all", min_face=settings.MIN_FACE,
                         cascade_path=settings.CASCADE)
    gray = to_gray(img)
    print(f"Photo: {path.name}   {img.shape[1]}x{img.shape[0]} px "
          f"(analysed at {gray.shape[1]}x{gray.shape[0]})")

    print("\nDetection passes (stops at the first that finds a face):")
    for i, (scale, neigh, single) in enumerate(DETECT_PASSES):
        n = len(clf.detector.detectMultiScale(gray, scale, neigh,
                                              minSize=(settings.MIN_FACE,) * 2))
        note = "  (only trusted if exactly 1)" if single else ""
        print(f"  pass {i + 1}: scale {scale} neighbors {neigh} -> {n} face(s){note}")

    boxes = detect_faces(clf.detector, gray, settings.MIN_FACE)
    if not boxes:
        print("\nRESULT: no_face. Even the most forgiving pass found nothing. "
              "The face may be too small (MIN_FACE), turned away, or covered.")
        return
    print(f"\nFaces used: {len(boxes)} (biggest first)")

    shown = img.copy()
    scale = img.shape[1] / gray.shape[1]
    for n, box in enumerate(boxes, 1):
        patch = face_patch(gray, box, clf.face_size)
        rank = ranking(clf, patch)
        best_name, best_dist = rank[0]
        ok = best_dist <= settings.THRESHOLD
        verdict = clean_name(best_name) if ok else "unknown_person"
        print(f"\nFace {n}: box x={box[0]} y={box[1]} size {box[2]}x{box[3]}")
        for name, d in rank[:5]:
            print(f"   {name:<24}{d:7.1f}")
        print(f"   threshold {settings.THRESHOLD}: closest is {best_name} at {best_dist:.1f} "
              f"-> {'MATCH' if ok else 'too far'} -> {verdict}")
        x, y, w, h = (int(v * scale) for v in box)
        cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(shown, f"{verdict} {best_dist:.0f}", (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    out = settings.OUTPUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", shown)
    if ok:
        buf.tofile(str(out / "explain.jpg"))
        print(f"\nAnnotated picture: {out / 'explain.jpg'}  (open it: is the green box on the face?)")


if __name__ == "__main__":
    main()
