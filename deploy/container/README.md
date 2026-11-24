# Container definition

`text_lab.def` is the Apptainer definition for the Text Lab image, and
`locks/` holds the package versions of its environments. The image
holds every runtime dependency of the app; nothing is installed on the host.

## Conda environments in the image

| Environment | Used for |
|---|---|
| `text_lab_main` | The Streamlit app, the FastAPI web UI and most features (default `python`) |
| `paddle_backend` | PaddleOCR worker |
| `paddle_vl_backend` | PaddleOCR-VL worker of the automatic OCR pipeline |
| `olmocr_backend` | olmOCR (vLLM) |
| `kraken_backend` | kraken line segmentation for historic handwriting (Torch 2.9) |
| `transcription_backend` | Transcription worker with WhisperX 3.8 and pyannote 4 (numpy 2) |
| `hf_ocr_backend` | Custom Hugging Face OCR models chosen by users (Transformers 5) |

Workers in the separate environments are started as subprocesses; their
interpreters are exported as `PADDLE_BACKEND_PYTHON`,
`PADDLE_VL_BACKEND_PYTHON`, `OLMOCR_BACKEND_PYTHON`, `KRAKEN_BACKEND_PYTHON`,
`TRANSCRIPTION_BACKEND_PYTHON` and `HF_OCR_BACKEND_PYTHON`.

The last three were added in November 2026 ahead of the code that uses them,
so one image build serves several feature branches. Until that code is
merged, the app does not start them: transcription still runs in
`text_lab_main` with WhisperX 3.7, which stays installed there, so the new
image also runs the current code.

Why the environments are split: the main environment is held at Torch 2.8
(torchaudio 2.9 removed APIs pyannote-audio 3 uses) and numpy 1.x
(WhisperX 3.7, unbabel-comet for Translation). Paddle has no CUDA 12.8
wheel, so both Paddle environments use its CUDA 12.6 build. olmOCR needs
vLLM 0.11.2 with Torch 2.9. kraken 7.1 needs Torch >= 2.9 and pins
`safetensors~=0.7`; Transformers 5 needs `safetensors>=0.8`; WhisperX 3.8
needs numpy >= 2.1. PaddleOCR-VL needs `paddlex[ocr]` installed explicitly,
or its pipeline fails at start with a DependencyError.

Models are not baked into the image. They live on research storage and are
bind-mounted at runtime under `/opt/...`; their locations are set in
`deploy/site.env`.

## Version locks

Unpinned packages drift between builds, and a drifted package can break a
feature without failing the build or an import test:

- 19 Nov 2026: 170 of 363 packages in `text_lab_main` moved, some to
  releases that need numpy 2 (pyarrow 26 broke Chat).
- 24 Nov 2026: `paddle_vl_backend` got sentencepiece 0.2.3, made for
  protobuf 7, next to protobuf 6.33, and PaddleOCR-VL failed at start.

So `locks/<env>.txt` pins every package of an environment:

| File | Environment | Contents |
|---|---|---|
| `locks/main.txt` | `text_lab_main` | production image `text_lab_210526.sif` |
| `locks/paddle.txt` | `paddle_backend` | production image |
| `locks/paddle_vl.txt` | `paddle_vl_backend` | production image |
| `locks/olmocr.txt` | `olmocr_backend` | production image |
| `locks/kraken.txt` | `kraken_backend` | Torch family only, not yet locked |
| `locks/transcription.txt` | `transcription_backend` | Torch family only, not yet locked |
| `locks/hf_ocr.txt` | `hf_ocr_backend` | Torch family only, not yet locked |

`main.txt` is used as a constraint: new packages (the web UI's fastapi, ...)
resolve within it. It leaves out numpy, protobuf and torchmetrics: the
unbabel-comet step needs numpy<2, protobuf<5 and torchmetrics<0.11 and
installs those itself, after streamlit and whisperx 3.7 have pulled in newer
ones. Pinning them makes the first install step unresolvable; `pip check`
reports the same conflicts in the production image. The Paddle and olmOCR
locks are installed as requirements and followed by `pip check`.

To upgrade a package, change its line in the lock and rebuild. To lock a new
environment once an image with it has been tested, write its `pip list
--format=freeze` into its lock file, without pip, setuptools and wheel. The
locks are copied into the image under `/opt/textlab/locks`, so an image
records what it was built with.

## Building

Build from a folder that holds `text_lab.def` and `locks/` (the def copies
`locks` by a relative path), on a compute node, never on a login node:

    cd deploy/container
    apptainer build /path/to/text_lab_DDMMYY.sif text_lab.def

Then copy the image to the container folder on research storage under a new
dated name, for example `text_lab_DDMMYY.sif`. Do not move or delete
existing images there: running sessions may still use them. Test the new
image in a sandbox app first by setting `TL_CONTAINER` in `template/dev.env`
(see the developer guide), then switch production to it.

The build needs no GPU. Before using a new image, run an import test of
every environment (`apptainer exec --nv`) and test OCR, Chat, Transcription
and Translation in the UI: import tests miss version clashes that only show
when a model loads. Things that failed builds before:

- `add-apt-repository` hung on the Launchpad API, so the Apptainer PPA is
  added by hand with a pinned key fingerprint.
- The PaddlePaddle package index was briefly missing the
  `paddlepaddle-gpu==3.3.1` wheel; if that happens, retry later.
- NLTK refuses data directories that others can write to; the def
  creates `/opt/nltk_data` with a strict umask.

To add a dependency, add it to `%post` in `text_lab.def` and rebuild. Keep
the pins: the image combines several GPU stacks (WhisperX, vLLM,
PaddlePaddle, Ollama) whose versions constrain each other.

Browser libraries of the web UI (htmx and its extensions) are not part of
the image. They are vendored in `src/textlab/ui/web/static/vendor/`, so the
app needs no network access and they change with the code, without a
rebuild.
