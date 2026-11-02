# Live Demo: Segmentation and Grad-CAM in the Browser

A Gradio app for the validated Hybrid-L15CA model. Upload a spinal endoscopy
frame, see the segmentation, and inspect Grad-CAM attention for any detected
structure at any of the five attention layers.

**Hosted:** [model-demo-mmeo.onrender.com](https://model-demo-mmeo.onrender.com)
(free Render instance; it sleeps when idle, so the first request after a pause
can take close to a minute).

This is a research prototype, not a medical device. Do not upload identifiable
patient data to the hosted instance: Gradio writes uploads to a temporary
folder on the server.

## What it does

- **Segmentation (live, ONNX):** every upload is segmented by `export/best.onnx`
  through ONNX Runtime on CPU, using a hand-written decode (no Ultralytics
  postprocessing).
- **Grad-CAM (PyTorch):** class-discriminative attention for the selected
  structure at L11, L15, L19, L23, or L27, computed from `weights/best.pt`
  (the seed 2 checkpoint, the same one that was exported).

The app picks one of two modes at startup:

| Mode | When | Grad-CAM |
|---|---|---|
| Live | PyTorch and the patched Ultralytics are installed (the full local setup) | Computed on demand for any frame |
| Gallery | They are not installed (the hosted build) | Precomputed for the 8 example frames in `demo/gallery/`; other uploads get segmentation only |

Set `GRADCAM_MODE=live` or `GRADCAM_MODE=gallery` to override the automatic
choice.

## Why two inference paths

An ONNX graph is forward-only. Grad-CAM needs the gradient of a class score with
respect to a layer's activations, which requires PyTorch autograd, so the
Grad-CAM path cannot use the ONNX file. Segmentation can, and running it through
ONNX Runtime keeps PyTorch out of the hosted build. This is a portability and
memory choice, not a speed one: Phase 5 found native `.pt` inference faster than
ONNX on this GPU (see [`export/README.md`](../export/README.md)).

## Segmentation pipeline (`onnx_infer.py`)

1. Shrink the upload so its longest side is at most 960 px.
2. Letterbox to 640 × 640 (pad value 114), convert BGR to RGB, scale to [0, 1].
3. Run ONNX Runtime. `output0` is 1 × 42 × 8400 (4 box values, 6 class scores,
   and 32 mask coefficients for each of the 8400 anchors across the three
   detection scales); `output1` is 1 × 32 × 160 × 160 (mask prototypes).
4. Keep anchors whose best class score is at least 0.25.
5. Greedy non-maximum suppression per class (IoU 0.45), written by hand.
6. For each surviving detection: mask = sigmoid(coefficients · prototypes),
   upsample to 640 × 640, remove the letterbox padding, resize to the image,
   crop to the detection's box, threshold at 0.5.
7. Map the boxes back to image coordinates.

The ONNX session runs on the CPU provider with one thread and the memory arena
disabled, which keeps the footprint small on the hosted instance.

**Verification against `.pt`.** `test_onnx_decode.py` runs both paths on one frame
and prints boxes, confidences, and the best mask IoU for each detection. On three
sample frames, boxes agree within about 6 px (1080p frames) and mask IoU is 0.92
to 0.99. Confidence scores can differ more: on one frame the Intervertebral Disc
score was 0.546 in ONNX and 0.422 in PyTorch, in line with the floating-point
differences already noted in `export/README.md`. The first version of the decode
skipped the final crop to the detection's box and scored 0.09 to 0.54 mask IoU;
Ultralytics' own mask postprocessing ends with that crop, and adding it closed
the gap.

## Grad-CAM (`gradcam_infer.py`)

The same procedure as `gradcam/run_gradcam.py`: a forward hook captures the
chosen layer, the backprop target is the highest raw (pre-NMS) score for the
selected structure, the gradient is averaged per channel to weight the
activations, then ReLU, crop to the content region (dropping the letterbox
padding), normalize, and overlay on the frame. The confidence shown in the status
box is that raw score. Method and verification are in
[`gradcam/README.md`](../gradcam/README.md).

| Layer | Mechanism |
|---|---|
| L11 | MSCA (backbone terminus) |
| L15 | Coordinate Attention (FPN fusion) |
| L19 | Triplet Attention (segmentation-head input, 80 × 80) |
| L23 | Triplet Attention (segmentation-head input, 40 × 40) |
| L27 | MSCA (segmentation-head input, 20 × 20) |

**L27 can show a flat heatmap.** Because the target is a single anchor, the
gradient reaches only the scale branch that anchor belongs to, and L27 receives
exactly zero gradient whenever the winning anchor comes from the 80 × 80 or
40 × 40 branch. The overlay is then a uniform tint with no hot spots. This is the
zero-gradient artifact described in `gradcam/README.md`, not a localization
result. In this gallery 6 of the 70 precomputed overlays show it (Skeleton 2,
Ligament 1, Nerve 1, Herniation 2), so it is not limited to the small structures.

## Using the demo

1. Pick an example frame from the dropdown, or upload your own.
2. Pick a structure. The list contains the structures detected in that frame.
3. Pick an attention layer. The Grad-CAM image and status box update.

| Input | Segmentation | Grad-CAM |
|---|---|---|
| Example frame | Live | Precomputed (hosted) or live (local) |
| Upload that is the same picture as a gallery frame | Live | Precomputed (hosted) or live (local) |
| Any other upload | Live | Live (local); a short message on the hosted instance |
| Image the model does not recognize | May return no detections; the status box says so | None |

An upload counts as a gallery frame when a 64 × 36 grayscale fingerprint of the
shrunk image is within a mean difference of 2.0 (on a 0 to 255 scale) of a gallery
frame's fingerprint. Re-saved or resized copies of the same picture match; crops
and edited versions do not. The matching frame then appears in the example
dropdown.

The model was trained only on spinal endoscopy frames from one institution. Other
images can return nothing or wrong structures.

## Contents of this folder

| File | Purpose | Needed by the hosted build |
|---|---|---|
| `app.py` | Gradio app: UI, mode selection, gallery matching | Yes |
| `config.py` | Paths, class names, thresholds, layer indices, image size limit | Yes |
| `onnx_infer.py` | ONNX Runtime session, preprocessing, decode, NMS, masks | Yes |
| `gradcam_infer.py` | Grad-CAM from the PyTorch checkpoint (loaded only in live mode) | No |
| `gallery/` | Precomputed example frames, Grad-CAM overlays, and `index.json` | Yes |
| `requirements.txt` | Hosted build dependencies (no PyTorch) | Yes |
| `requirements_live.txt` | Dependencies for live mode on a host with more memory | No |
| `make_gradcam_gallery.py` | Regenerates `gallery/` from a folder of images | No |
| `test_onnx_decode.py` | Checks the ONNX decode against the PyTorch checkpoint | No |

`weights/best.pt` and `export/best.onnx` are tracked even though `.gitignore`
excludes `*.pt` and `*.onnx`. If you replace either file, git keeps tracking the
existing path; a new model file needs `git add -f`.

## Run locally

Full setup (live Grad-CAM), from the repository root:

```bash
pip install -r requirements.txt
python scripts/apply_integration.py   # registers the custom attention modules
python demo/app.py
```

Open `http://localhost:7860`. The console prints `http://0.0.0.0:7860`, which
browsers cannot open on Windows.

To preview the hosted behavior without removing PyTorch:

```powershell
$env:GRADCAM_MODE="gallery"; python demo/app.py     # PowerShell
```
```bash
GRADCAM_MODE=gallery python demo/app.py             # bash
```

To run the lean hosted environment, install `demo/requirements.txt` instead; the
app then starts in gallery mode. Always run from the repository root, because the
model paths in `config.py` are relative to it.

To check the ONNX decode against the checkpoint (needs the full setup; it reads the frame named in `TEST_IMAGE` at the top of the file, `data/frames/10-1_10_Video2_00039.jpg` by default):

```bash
python demo/test_onnx_decode.py
```

## Regenerate the gallery

```bash
python demo/make_gradcam_gallery.py --source <folder of validation images>
```

Options: `--per-class` (frames per structure, default 2), `--max-scan` (most images
to examine, default 600), `--seed` (default 0). It needs the full setup because it
calls the same `gradcam_for_layer` function as live mode.

Frames are drawn at random with a fixed seed until each of the six structures is
detected (score 0.5 or higher) in `--per-class` frames, so examples are not chosen
by model confidence. For each frame the script saves a shrunk JPEG, then runs
segmentation on that saved JPEG and writes one Grad-CAM overlay per detected
structure and layer, plus `index.json`. The app therefore sees exactly the pixels
the overlays were computed from. The script replaces the existing `gallery/`
folder, so commit the result (`git add demo/gallery`).

The committed gallery has 8 frames covering all six structures: 14
frame-structure pairs × 5 layers = 70 overlays, about 4.8 MB. The frames are
shown with permission under the institutional agreement covering the dataset; the
dataset itself is not distributed.

## Deployment (Render)

| Setting | Value |
|---|---|
| Runtime | Python 3, with `PYTHON_VERSION=3.12.7` |
| Root Directory | empty (repository root) |
| Build Command | `pip install -r demo/requirements.txt` |
| Start Command | `python demo/app.py` |
| Instance | Free (512 MB) |

The app binds to `0.0.0.0` and the port Render provides in `PORT`. Root Directory
must stay empty because `export/best.onnx` and the gallery are located relative to
the repository root.

**Why the hosted build has no PyTorch.** An earlier build that loaded PyTorch and
Ultralytics at startup was stopped by Render for exceeding the free instance's
memory (the Events tab reported "Ran out of memory (used over 512MB)"), and live
Grad-CAM never returned there. The hosted build therefore installs only Gradio,
ONNX Runtime (CPU), OpenCV (headless), and NumPy, and serves precomputed Grad-CAM
for the example frames. The 512 MB limit is a constraint to plan around; the peak
memory of the current hosted build was not measured on Render.

**Upgrade path.** On an instance with more memory, install
`demo/requirements_live.txt` and run the integration patch in the build step:

```
pip install -r demo/requirements_live.txt && pip install --force-reinstall --no-deps opencv-python-headless==4.11.0.86 && python scripts/apply_integration.py
```

The app then starts in live mode with no code change. The final command reinstalls
the headless OpenCV build last, because Ultralytics also installs the regular
build, which needs display libraries that servers often lack. `PYTHON_VERSION`
must stay at 3.12.x: the pinned PyTorch 2.5.1 has no wheels for newer Python
versions, and the integration patch is a copy of Ultralytics 8.3.185 files, so
that version is pinned. This build completed on the free instance, but it has not
been run on a larger one, so live Grad-CAM there is untested.

## API and tests

Gradio exposes the app over REST. The segmentation sequence is three calls:

| Step | Request | Result |
|---|---|---|
| 1 | `POST /gradio_api/upload` (form-data, key `files`) | Server-side file path |
| 2 | `POST /gradio_api/call/segment`, body `{"data": [{"path": "<path from step 1>", "meta": {"_type": "gradio.FileData"}}]}` | `event_id` |
| 3 | `GET /gradio_api/call/segment/{event_id}` | Server-sent event stream ending in `event: complete` |

The `complete` event carries a four-item array: the overlay image, a dropdown
update whose `choices` list the detected structures, and two internal state
placeholders. `GET /` and `GET /config` serve as health and configuration checks;
`/config` includes the Gradio version.

`use_example` (a gallery frame ID in; the frame, its segmentation overlay, and the
structure list out) is also exposed but is not covered by the tests below. `run_gradcam` cannot be used on its own: its
Structure input is validated against the choices set by the previous step, which
are empty in a fresh session, so a direct call returns `Value: Nerve is not in the
list of choices: []`. Grad-CAM therefore needs a browser session and is outside
the API tests.

### Postman collection

`tests/postman/surgical_yolo_demo.postman_collection.json` contains the requests
below, each with its checks in the Post-response script. Collection variables:
`base_url`, plus `file_path`, `event_id`, and `bad_path`, which the scripts set.

| Request | What it sends | Checks |
|---|---|---|
| 1 Health check | `GET /` | Status 200 |
| 2 Config | `GET /config` | Status 200; `version` is a string |
| 3 Upload | Sample frame as form-data | Status 200; returns a file path (stored in `file_path`) |
| 4 Run segmentation | `POST /gradio_api/call/segment` | Status 200; returns an `event_id` |
| 5 Get result | `GET /gradio_api/call/segment/{event_id}` | Stream has `event: complete`; `data[0].url` is a string; `data[1].choices` is not empty |
| N1 Missing data field | Body `{}` | Status 422 |
| N2 No image | Body `{"data": [null]}` | Call returns 200; the stream completes with a null image and no structures |
| N3 Non-image file | A text file named `fake.jpg`, uploaded and segmented | Upload returns 200; the stream contains `event: error` while the HTTP status stays 200 |
| N4 Unknown endpoint | `POST /gradio_api/call/nope` | Status 404 or 500 |

**Running it.**

1. Start the app locally, or use the hosted URL.
2. In Postman, import the collection file and set `base_url` to
   `http://localhost:7860` or `https://model-demo-mmeo.onrender.com` (no trailing
   slash).
3. In request 3, select `data/frames/9-1_Video5_24263.jpg` in the form-data body.
   In N3, select any text file renamed `fake.jpg`.
4. Send 1, 2, 3, 4, 5, then N1, N2, N3, N4, in that order. Send request 4 again
   before every request 5.

The main flow was run against the hosted instance and the whole collection passes.

**Notes from building the tests.**

- Set every JSON body to Body > raw > JSON. With the raw type left on Text, Postman
  sends `text/plain` and Gradio answers 422 even for a valid body.
- An event ID can be read once. A second `GET` on the same ID does not fail; the
  stream sends heartbeats until the client times out. This is why request 4 is sent
  before each request 5.
- Type request URLs by hand. A pasted URL carried an invisible trailing tab
  (`%09` in the Postman Console) that turned a valid event ID into a 404.
- On Windows, backslashes in the uploaded file path break the JSON body. The
  request 3 script escapes them.
- The version check only asserts that `version` is a string, because the local
  Gradio was 6.29.0 while `requirements.txt` pins 6.29.1.
- Postman shows the event stream only after it closes, so allow extra time on the
  free instance.

**Findings.**

- A non-image file passes the upload and fails at segmentation. The failure is
  reported inside the stream as `event: error` while the HTTP status stays 200, so
  a client must read the stream, not just the status code.
- Running the collection against the earlier PyTorch-loading build returned
  `404: Not Found` for the result request after Render restarted the instance
  (the Events tab showed the out-of-memory stop), which wipes Gradio's in-memory
  event list. This is how the API tests exposed the hosting limit described above.

**Not covered:** Grad-CAM, segmentation accuracy, load or latency, more than one
sample frame, and authentication (the app has none).

## Known limitations

- On the hosted instance Grad-CAM exists only for the 8 example frames. Live
  Grad-CAM for any frame needs a local install with PyTorch.
- The gallery is small (8 frames). It illustrates the method and is not a
  statistical sample.
- Gallery matching recognizes the same picture only; crops and edits are treated
  as new images.
- The free instance sleeps when idle and has 512 MB of memory.
- Results come from one model trained on one institution's data, as described in
  the main [README](../README.md#limitations).