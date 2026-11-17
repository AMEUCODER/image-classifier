# Image sorter: face classifier

Sorts photos into folders by **who is in them**, using OpenCV (Haar cascade finds faces, LBPH recognizes them). Part of a larger image-sorter project where each classifier (faces, objects, scenes, month/year) plugs into the same pipeline.

```
pictures --> scan --> classifier --> label --> copy to output/faces/<label>/ --> sort_log.csv (DataFrame)
```

## Setup

```
pip install -r requirements.txt
```
It needs `opencv-contrib-python` (not plain `opencv-python`): the `cv2.face` module lives in the contrib build.

## Run it

No arguments. Open `main.py`, edit the SETTINGS block at the top if needed, then:

```
python main.py
```
(or press Run in your editor). By default it sorts every picture in `Faces/val/` into `output/faces/<person>/`.

On every run it does two things:
1. **Trains anyone new.** Every person folder in `Faces/train/` that the model doesn't know yet is trained. First run: everyone. Later runs: only new folders.
2. **Sorts the pictures** in `INPUT_FOLDER` and writes `output/sort_log.csv`.

## Add a new person

1. Create `Faces/train/<their name>/` and put 20 to 50 clear photos in it.
2. Run `python main.py`.

That's it: only the new person is trained, the others are untouched, and then they are recognized. (Case, spaces and underscores in the folder name don't matter.)

## Test it with your own people (for example Madonna)

No photos of real people are included, so test with yours (your course folders `Faces/train` and `Faces/val` work as they are):

1. `Faces/train/Madonna/` has 20 to 50 photos. Do the same for one or two more people.
2. `Faces/val/Madonna/` has 5 to 10 **different** photos of her. Add one val folder for a person who is **not** in train: that acts as a stranger.
3. `python tools/check_dataset.py` tells you which photos are unusable (no face, several faces, unreadable), which people have too few photos, and whether the same picture sits in both train and val (which fakes good accuracy).
4. `python main.py` trains and sorts `Faces/val` into `output/faces/<person>/`.
5. `python tools/evaluate.py` compares every val photo with its folder name: overall accuracy, accuracy per person, and a threshold sweep that suggests a `THRESHOLD` (needs a stranger folder to do that).

## A photo came out wrong? Use `explain.py`

```
python tools/explain.py "C:\path\to\photo.jpg"
```
It shows whether a face was found and by which detection pass, the distance to **every** trained person, the threshold, and the verdict, and saves `output/explain.jpg` with the detected face boxed so you can check the box is on the face.

| What you see | Meaning | What to do |
|---|---|---|
| `no_face` | the detector found nothing | the face is small, turned away or covered; lower `MIN_FACE`, or use a clearer photo |
| `unknown_person`, closest is the right person but "too far" | threshold too strict | raise `THRESHOLD` (see the distances printed) |
| `unknown_person`, closest is the wrong person or far from everyone | not enough or too different training photos | add 30 to 50 varied photos of that person and retrain |
| a name that is wrong | too few training photos for one of the people | add photos; consider embeddings (see limits) |

Training prints how many photos each person actually used, for example `trained Madonna: 9 of 12 photos used`, plus every skipped photo and why. If many are skipped, fix the photos first.

## Settings (top of `main.py`)

| Setting | Meaning |
|---|---|
| `INPUT_FOLDER` | pictures to sort, searched recursively (default `Faces/val`) |
| `OUTPUT_FOLDER` | where sorted copies and `sort_log.csv` go (default `output`) |
| `TRAIN_FOLDER` | one sub-folder per person (default `Faces/train`) |
| `THRESHOLD` | max LBPH **distance** for a match. Lower = stricter. 80 is only a starting point |
| `POLICY` | several faces in a photo: `all` (copy to each recognized person), `largest`, `group` |
| `AUTO_TRAIN` | train new people before sorting |
| `DRY_RUN` | `True` = show what would happen, copy nothing. Good for a first look |
| `MOVE` | `True` = move files instead of copying |

Originals are copied, never moved, unless `MOVE = True`. Running again does not create duplicates: identical files already in the output are reused.

## Where pictures end up

| Folder | Meaning |
|---|---|
| `output/faces/<name>/` | recognized person |
| `output/faces/unknown_person/` | a face was found but the distance was over the threshold |
| `output/faces/no_face/` | no face detected |
| `output/faces/group/` | several faces, with `POLICY = "group"` |
| `output/_errors/` | file could not be read as an image |

### Tuning the threshold
1. Sort some photos of people you trained on, and some strangers.
2. Open `output/sort_log.csv` and compare the `distance` column for the two groups.
3. Set `THRESHOLD` between them. If they overlap, choose strict (more `unknown_person`) or loose (more wrong names).

## Project layout

```
main.py                  settings + run (no arguments)
tools/
  check_dataset.py       audit your photos before training
  evaluate.py            accuracy + threshold sweep against Faces/val
  explain.py             what happens to ONE photo, step by step (use this first when a photo looks wrong)
imagesorter/
  pipeline.py            scan, classify, copy, log (pandas DataFrame)
  classifiers/
    base.py              shared contract: Result + BaseClassifier
    faces.py             detection, recognition, training, auto-sync
  imgio.py               image loading (works with non-ASCII Windows paths)
models/
  haar_face.xml          face detector (included)
  faces_trained.yml      created on first run
  faces_labels.json      created on first run (names + crop size: keep the pair together)
Faces/train/             your training photos, one folder per person
Faces/val/               pictures to sort (default input)
scripts/
  face_recognize_auto.py your original single-file script, upgraded: add a name to the
                         END of its `people` list + photos in Faces/train/<name>
tests/                   pytest suite (real face detection + synthetic logic)
docs/                    faces.md notes, classifiers.base (Obsidian Bases view)
```

Run the tests: `pip install -r requirements-dev.txt`, then `python -m pytest -q`.

## Using your existing model (`face_trained.yml`)

This project makes its own files (`faces_trained.yml` + `faces_labels.json`). Your old model has no labels file and was trained without the histogram equalization used here, so **don't mix them**. On the first run `main.py` simply trains a fresh model from your `Faces/train/` photos. To keep using the old model, use `scripts/face_recognize_auto.py` instead.

## Updating from an older version

If `models/` holds files made by an older version of this code, `python main.py` notices, keeps them as `*.old`, and retrains from your `Faces/train` photos.

## Retrain from scratch

`main.py` only trains people the model does not know yet. To rebuild everything (for example after adding photos to an existing person), delete `models/faces_trained.yml` and `models/faces_labels.json`, then run `python main.py`.

## Known limits

- LBPH is sensitive to lighting and head angle; expect some `unknown_person` at first.
- Haar misses tilted or side-facing faces; those land in `no_face`. Detection uses your original script's settings first, then one more forgiving pass that is only trusted when it finds exactly one face (measured on test pictures: finds more faces without inventing faces in non-face photos).
- Each face is cropped, resized and contrast-normalized on its own, so a dark or bright background does not change how it is compared.
- Photos in a person's folder with zero or several detected faces are skipped (the run prints each one and why).
- Upgrade path: swap in face embeddings (FaceNet / `face_recognition`) behind the same `classify(image)` interface. Nothing else changes.
- Big photos are shrunk to 1200 px on the long side before detection, so very small faces in huge photos can be missed.
- Privacy: don't publish photos of real people. `.gitignore` already excludes your training and val photos and the trained models.

## License

MIT, see `LICENSE` (put your name in it). The Haar cascade in `models/` is OpenCV's and keeps its own license.
