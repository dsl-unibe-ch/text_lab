# Container definition

`text_lab.def` is the Apptainer definition for the Text Lab image. The image
holds every runtime dependency of the app; nothing is installed on the host.

## Conda environments in the image

| Environment | Used for |
|---|---|
| `text_lab_main` | The Streamlit app and most features (default `python`) |
| `paddle_backend` | PaddleOCR worker |
| `paddle_vl_backend` | PaddleOCR-VL worker of the automatic OCR pipeline |
| `olmocr_backend` | olmOCR (vLLM) |

Workers in the separate environments are started as subprocesses; their
interpreters are exported as `PADDLE_BACKEND_PYTHON`,
`PADDLE_VL_BACKEND_PYTHON` and `OLMOCR_BACKEND_PYTHON`.

Models are not baked into the image. They live on research storage and are
bind-mounted at runtime under `/opt/...` (see `template/script.sh.erb`).

## Building

Build on a compute node, never on a login node, then copy the image to the
container folder on research storage under a new dated name, for example
`text_lab_DDMMYY.sif`. Do not move or delete existing images there: running
sessions may still use them. Point `template/script.sh.erb` (or, once it
exists, the site configuration) at the new image and test it in the sandbox
app before switching production to it.

To add a dependency, edit the `%post` section of `text_lab.def` and rebuild.
Keep the version pins: the image combines several GPU stacks (WhisperX,
vLLM, PaddlePaddle, Ollama) whose versions constrain each other.
