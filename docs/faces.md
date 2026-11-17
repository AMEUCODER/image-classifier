---
classifier: faces
project: image-sorter
status: in-progress
model: LBPH + Haar cascade
tags:
  - image-sorter
  - classifier
  - faces
updated: 2026-10-31
---

# Faces classifier

Part of the **image sorter** project: one pipeline, several swappable classifiers (faces, objects, scenes, month/year). This note collects everything decided and built for the faces classifier.

## 1. What it does

Image in, one or more face labels out. The label becomes the output folder name: `output/faces/<person_name>/`, plus `no_face/`, `unknown_person/` and optionally `group/`.

Face sorting is two separate problems:

- **Detection** finds where the faces are (Haar cascade).
- **Recognition** decides whose face it is (LBPH trained model).

## 2. Files and what each one is for

| File | Purpose |
|---|---|
| `haar_face.xml` / `haarcascade_frontalface_default.xml` | Detects faces. Ships with OpenCV (`cv2.data.haarcascades`), so it does not have to live in `models/` |
| `face_trained.yml` (old) / `faces_trained.yml` (package) | The trained LBPH model: says whose face it is |
| `faces_labels.json` | Names and the crop size used in training. The `.yml` stores numeric IDs only, so model and labels are a pair: a model with the wrong labels file silently gives wrong names |
| `base.py` | Shared contract: `Result(label, confidence, details)` and `BaseClassifier.classify(image)` |
| `faces.py` | Package version: classify + train + add a person |
| `face_recognize_auto.py` | Your original script, tuned to auto-train new people in the `people` list |

Needs `opencv-contrib-python` (`cv2.face` is in the contrib build).

## 3. Package layout (decided)

A package is the right call. The pipeline and dataframe are the centre of the project, not an afterthought.

```
image_sorter/
  main.py                      entry point
  models/                      trained .yml, labels, YOLO / Places files
  imagesorter/
    pipeline.py                scan, classify, copy files, log
    ui.py                      tkinter launcher (built last)
    classifiers/
      base.py
      faces.py
      objects.py               later
      scenes.py                later
      dates.py                 later (month/year from EXIF, fallback file date)
```

Dataframe idea: one row per image, one column per classifier (`faces`, `objects`, `scene`, `month_year`). A separate step decides which column(s) choose the folder. That gives combined sorting, a CSV log and a table for the interface.

## 4. Training steps (the workflow)

**A. Prepare the data**
1. One folder per person, named as you want the output folder named.
2. 20 to 50 photos each, with different lighting, angles and expressions.
3. Delete bad photos: sunglasses, tiny faces, blur, groups where you cannot tell who is who.
4. Hold out about 20% for testing and never train on them.

**B. Detect and crop**
1. Run the detector on every training photo.
2. Keep only photos with exactly one face.
3. Crop, convert to grayscale, resize to one fixed size (for example 100x100). Use the same size when sorting.

**C. Train and save**
1. Give each person a numeric ID and keep the ID-to-name mapping.
2. Train the recognizer on crops and IDs.
3. Save the model and the labels file together.

**D. Tune the threshold**
1. Run on held-out photos and record distances for correct matches.
2. Run on strangers and record those distances.
3. Pick a value between the two groups. If they overlap, choose between missing people (strict) and mislabeling them (loose).
4. Keep the value in a config file, not hardcoded.

**E. Test in isolation**
1. A small folder of about 15 photos: known people, strangers, no-face images, a group photo.
2. Check each lands where expected.
3. Test each multi-face policy separately.

**F. Connect to the pipeline**
The classifier only returns labels. The pipeline loads the image, calls it, and copies the file into `output/faces/<label>/`.

## 5. Running it (zip version)

Plain `python main.py`, no arguments. Settings sit at the top of `main.py` (`INPUT_FOLDER`, `OUTPUT_FOLDER`, `TRAIN_FOLDER`, `THRESHOLD`, `POLICY`, `DRY_RUN`, `MOVE`). Each run trains any new person folder found in `Faces/train/` (everyone on the first run), then sorts `INPUT_FOLDER` into `output/faces/<person>/` and writes `sort_log.csv`. Re-running creates no duplicates.

To add a person: create `Faces/train/<name>/` with photos and run again.

Tools: `python tools/explain.py <photo>` shows what happens to one photo (detection passes, distance to every person, verdict, annotated image); `python tools/check_dataset.py` audits the photos (unusable ones, too few photos, train/val leaks); `python tools/evaluate.py` measures accuracy on `Faces/val` and suggests a threshold.

### Package commands (optional)

Dataset layout (folder name = person name = output folder name):

```
dataset/
  ben_affleck/  img1.jpg img2.jpg ...
  madonna/      ...
```

Train from scratch:

```
python -m imagesorter.classifiers.faces train dataset models/
```

Add one person without retraining the others (if the name exists, the photos are added to that person):

```
python -m imagesorter.classifiers.faces add new_person_photos models/ --name jane_doe
```

From Python: `FaceClassifier.train(...)`, `FaceClassifier.add_person(...)`, and `FaceClassifier(model, labels)` to classify. Module-level aliases `train_model` and `add_person` exist because `classifiers/__init__.py` exports them.

Training details:
- Only photos with exactly one detected face are used; every skipped photo is printed with the reason.
- Each face is also added mirrored (doubles the data). Turn off with `augment=False`.
- `--no-detect` treats images as already-cropped faces.
- The crop size is stored in the labels file and read back by the classifier, which removes training/sorting size mismatch.
- Old plain lists (`.json` or `.txt`) of names still load, but then set `face_size` yourself.

## 6. Multi-face policy

| Policy | Behaviour |
|---|---|
| `all` | One label per recognized person; the image is copied to each person's folder |
| `largest` | Only the biggest face counts |
| `group` | More than one face gives the single label `group` |

Fallbacks: no face gives `no_face`; distance over the threshold gives `unknown_person`.

An optional variant skips recognition and sorts by face count only: `no_people`, `one_person`, `group`.

## 7. Your script, tuned (`face_recognize_auto.py`)

**How to add a person**
1. Add the name at the **end** of `people`.
2. Put photos in `../Faces/train/<name>/` (case, spaces and underscores do not matter).
3. Run the script. It trains only the new person, saves the model, then recognizes.

**How it works:** it asks the loaded model which IDs it already knows (`getLabels()`) and trains only the missing ones with OpenCV's `update()`. A `face_trained.yml.bak` copy is made before each save. The first run on your existing model should train nothing.

**Rules that matter**
- **Only add at the end of the list.** A person's position is their ID inside the model. Inserting or reordering gives everyone the wrong name.
- **Keep the same preprocessing.** The script uses a plain grayscale crop, no resize and no `equalizeHist`, to match how your existing model was trained. The package `faces.py` equalizes the histogram, which suits models trained from scratch with it. Do not mix it with the old `.yml`.
- **Strangers get a name.** `THRESHOLD = None` means the closest person always wins. Look at printed distances and set a number to get `Unknown`.
- Paths are resolved relative to the script (`../models`, `../Faces/train`), so it works from any run folder. Change `TRAIN_DIR` if your folder is named differently.
- For new people it uses the largest face per photo, so stray false detections do not pollute training.

## 8. Haar cascade and the `.yml`: yes, for version 1

Both are used because they match what you already have. Two checks before plugging in the old model:

1. A labels list in the same order as training (ID 0 is the first name).
2. The crop size used in training (if it never resized, leave it empty).

Your current model only knows the 5 celebrities (Ben Afflek, Elton John, Jerry Seinfield, Madonna, Mindy Kaling). Anyone else comes out as the nearest of them unless a threshold is set.

## 9. Known limits and the upgrade path

- LBPH is sensitive to lighting and head angle, so expect a lot of `unknown_person` at first.
- Bug found in real use (v1): whole-image histogram equalization before detection missed faces on dark or bright backgrounds (3 of 13 test cases) that the original script found. Fixed: plain detection first, a single-face-only forgiving pass second, per-face normalization. Models from v1 are retrained automatically.
- Haar misses tilted and side-facing faces. If photos clearly containing faces end up in `no_face`, swap the detector (MediaPipe or a DNN detector). Only the detection step changes.
- Upgrade path for recognition: face embeddings (the `face_recognition` library, FaceNet or ArcFace) with cosine or Euclidean distance. The classifier interface stays the same, so the rest of the project does not change. Worth showing in the README as a design decision.

## 10. Testing status

- Logic tested on synthetic images with a fake detector: training, recognizing two people, rejecting a stranger, adding a third person, older people still recognized after the add, name with no folder skipped.
- The tuned script was tested the same way: first run trains everyone, second run trains nothing, a person added to the list is trained and recognized, a backup is written.
- **Not tested with real Haar detection**: the sandbox OpenCV had no cascade file (environment issue, a normal install includes it). Try it on your own photos.

## 11. Portfolio notes for the faces part

- The image sorter gets its **own repo**, with the classifiers as a package inside it. A repo with only classifier files hides the data flow, which is what proves the skills.
- Show: the results dataframe and an analysis report (pandas), the shared base class and train/evaluate/tune workflow with real numbers (classifiers), and a Mermaid data-flow diagram in the README (data flow).
- **Do not commit private faces.** Use a public dataset such as LFW and link it, or commit a tiny sample. Web celebrity photos carry copyright.
- Do not commit big model files directly: use Releases or Git LFS, or keep the training script so anyone can rebuild them.
- Be honest about the limits (LBPH, lighting) in the README.

## 12. Build order and next steps

1. Pipeline only, with a fake classifier (tests files, folders, logging).
2. Face count mode.
3. Object mode with a YOLO-style detector.
4. Face recognition with the model above.
5. Scene mode with Places365.
6. Colour-theme mode with HSV.
7. tkinter launcher with a mode dropdown and progress bar.
8. Combined mode and a JSON config file.

Open questions:
- [ ] Real accuracy of the model on held-out photos
- [ ] Final threshold value
- [ ] Which multi-face policy to default to
- [ ] Whether to move to embeddings if LBPH accuracy disappoints
