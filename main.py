"""Face sorter. Edit the SETTINGS below, then just run it:

    python main.py          (or press Run in your editor, no arguments needed)

What happens on every run:
  1. Any person folder in Faces/train that the model does not know yet is trained
     (first run: everyone is trained). To add a person: make a folder
     Faces/train/<name>/ with their photos and run again.
  2. Every picture in INPUT_FOLDER is sorted into OUTPUT_FOLDER/faces/<person>/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from imagesorter.classifiers.faces import FaceClassifier, sync_dataset  # noqa: E402
from imagesorter.pipeline import sort_images, summary                   # noqa: E402

# ===================== SETTINGS (edit these) =====================
INPUT_FOLDER = ROOT / "Faces" / "val"      # pictures to sort (searched recursively)
OUTPUT_FOLDER = ROOT / "output"            # sorted copies + sort_log.csv go here
TRAIN_FOLDER = ROOT / "Faces" / "train"    # one sub-folder per person
MODELS_FOLDER = ROOT / "models"

THRESHOLD = 80.0     # max LBPH distance to accept a match (lower = stricter)
POLICY = "all"       # several faces in one photo: "all", "largest" or "group"
AUTO_TRAIN = True    # train new people found in TRAIN_FOLDER before sorting
DRY_RUN = False      # True = only show what would happen, copy nothing
MOVE = False         # True = move files instead of copying them

FACE_SIZE = 100      # training crop size (stored in the labels file)
MIN_FACE = 40        # smallest face in pixels the detector accepts
# =================================================================

CASCADE = MODELS_FOLDER / "haar_face.xml"
MODEL = MODELS_FOLDER / "faces_trained.yml"
LABELS = MODELS_FOLDER / "faces_labels.json"


def train_new_people():
    try:
        report = sync_dataset(TRAIN_FOLDER, MODEL, LABELS,
                              face_size=(FACE_SIZE, FACE_SIZE), min_face=MIN_FACE,
                              cascade_path=CASCADE)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"  {e}")
        return
    for note in report.get("notes", []):
        print(f"  NOTE: {note}")
    for name, n in report["counts"].items():
        used, total = report["photos"][name]
        print(f"  trained {name}: {used} of {total} photos used")
    for name, items in report["skipped"].items():
        for fname, why in items:
            print(f"  skipped {name}/{fname}: {why}")
    if not report["counts"]:
        print("  model is up to date, nobody new to train "
              "(python tools/check_dataset.py shows unusable photos)")


def main():
    if not CASCADE.is_file():
        sys.exit(f"Missing face detector: {CASCADE}")
    if not INPUT_FOLDER.is_dir():
        sys.exit(f"Input folder not found: {INPUT_FOLDER}\n"
                 f"Change INPUT_FOLDER at the top of main.py")

    if AUTO_TRAIN:
        print("Checking training folder...")
        train_new_people()
    if not (MODEL.is_file() and LABELS.is_file()):
        sys.exit(f"No trained model at {MODEL}\n"
                 f"Put photos in {TRAIN_FOLDER}/<person name>/ and set AUTO_TRAIN = True")

    classifier = FaceClassifier(MODEL, LABELS, threshold=THRESHOLD, policy=POLICY,
                                min_face=MIN_FACE, cascade_path=CASCADE)

    def progress(done, total, path):
        print(f"\r[{done}/{total}] {path.name[:40]:<40}", end="", flush=True)

    print("Sorting pictures...")
    df = sort_images(INPUT_FOLDER, OUTPUT_FOLDER, classifier,
                     dry_run=DRY_RUN, move=MOVE, progress=progress)
    print("\n")
    print(summary(df))
    if DRY_RUN:
        print("\nDRY_RUN is on: nothing was copied.")
    else:
        print(f"\nSorted into: {OUTPUT_FOLDER / 'faces'}")
        print(f"Log:         {OUTPUT_FOLDER / 'sort_log.csv'}")


if __name__ == "__main__":
    main()
