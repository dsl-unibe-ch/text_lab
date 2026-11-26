# Hugging Face OCR models

Reads documents with a vision-language model the user names on the
Hugging Face Hub. The model reads each page as a whole and returns its
text. It is separate from the OCR feature and from historic handwriting;
the three share no code apart from `common/`.

Users reach it on the OCR page, in the **Hugging Face OCR models**
expander below historic handwriting. The UI is in `ui/streamlit/hf_ocr/`
(`panel.hf_ocr_expander`).

## Pipeline

`service.read_document` (one PDF or image) and `service.read_batch` (a ZIP
archive) run the same steps. A batch sends all pages of all files through
one worker, so the model is loaded once per batch.

1. **Pages** (`common.pages.page_images`): a PDF is rendered at 200 DPI
   (`service.DPI`), every frame of an image file becomes an RGB PNG.
2. **Reading** (`worker.py`), in a worker in the image's `hf_ocr_backend`
   environment (Transformers 5, Torch 2.10):
    1. `check_model` asks the Hub about the model, without a token. It
       refuses a private, missing or gated model, a repository that holds
       only a LoRA adapter (`adapter_config.json` without `config.json`),
       and a model whose weights (2 bytes per parameter, from the Hub's
       safetensors metadata) would take more than 85% of the free GPU
       memory. Nothing has been downloaded at this point.
    2. `load_model` loads the processor and `AutoModelForImageTextToText`
       with `trust_remote_code=False` and `token=False`, downloading into
       the user's folder (`store.models_dir()`). A model Transformers
       cannot load this way is refused.
    3. `read_page` reads every page: scaled to at most 2048 pixels on its
       longest side, given to the processor with the prompt (as a chat
       message when the processor has a chat template, directly when it
       has none), and generated greedily up to the token limit. Only the
       generated part is decoded.
3. **Exports**: per document a `.txt` and a `.json` (pages, model,
   revision, prompt), and `model_used.txt` for citation.

The code is the same for every model; nothing depends on which one it
is. Errors meant for users are `worker.OcrModelError` in the worker; the
service turns them into `HfOcrError` with the same message, so the page
shows them without a traceback. The worker exits when it is done, so the
GPU is free afterwards; the panel first calls
`gpu_manager.prepare_gpu(HF_OCR)`.

## The user's model folder

Models do not go into the shared model store, which only the maintainers
fill. Each user's go into `~/.cache/textlab/hf_ocr_models` (or
`TEXT_LAB_HF_OCR_DIR`), laid out as a Hugging Face cache; the worker also
points `HF_XET_CACHE` there, so a download leaves nothing in the shared
store. `store.stored_models` lists the models with their size and
`store.delete_model` removes one; the panel offers both, since the folder
counts towards the user's quota.

## Public API

| Name | Module | Purpose |
|---|---|---|
| `read_document(name, data, options, on_progress=)` | `service` | Read one PDF or image; returns an `OcrRun` |
| `read_batch(archive, options, on_progress=)` | `service` | Read a ZIP archive; returns a `BatchRun` |
| `read_pages(images, options, ...)` | `service` | Run the worker on page images |
| `HfOcrOptions(repo_id, prompt, max_new_tokens)` | `models` | What to run; checks the values |
| `parse_repo_id(text)` | `models` | `owner/name` from a name or a Hub link |
| `HfOcrError` | `models` | A run could not be done, with a message for users |
| `stored_models()`, `delete_model(repo_id)`, `models_dir()` | `store` | The user's model folder |

## Layout

```
hf_ocr/
├── service.py   # the API above: pages, worker, outputs
├── models.py    # options, results, model names
├── store.py     # the user's model folder
├── worker.py    # model checks, loading and reading (hf_ocr_backend)
└── tests/
```

## Files written

| What | Where | Removed |
|---|---|---|
| The upload, page images, result files | `hf_ocr/run-*` in the job workspace | When the run ends |
| A batch's files, page images and result folders | `hf_ocr/batch-*` | When the batch ends |
| The worker's request, progress and result | `hf_ocr/job-*` | When the worker ends |
| The chosen models | `~/.cache/textlab/hf_ocr_models` | When the user deletes them |

Results are returned in memory. No document is written to the user's
home directory, only the models.

## Configuration

| Setting | Used for |
|---|---|
| `TEXT_LAB_HF_OCR_DIR` | Another folder for the user's models; optional |
| `HF_OCR_BACKEND_PYTHON` | Another interpreter for the worker; optional |

## Tests

In `tests/`, without GPU, network or models:

- `test_models.py`: model names and links, options, citation and JSON.
- `test_store.py`: the folder, sizes counted once, deleting.
- `test_worker.py`: what the worker may import; the size, gated and
  missing-model checks against a fake Hub; reading a page with a fake
  model and processor, with and without a chat template.
- `test_service.py`: the service with a fake worker: outputs, batch
  layout, errors for users, the interpreter.
- `test_panel.py`: the panel uses only the service and the store.

## Possible extensions

- Packages some models ask for (`sentencepiece`, `tiktoken`, `einops`,
  `timm`, ...) are not in `hf_ocr_backend` yet; adding them needs an image
  rebuild (`deploy/container/text_lab.def`).
- Gated models would need the user's own Hugging Face token.
