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
bind-mounted at runtime under `/opt/...`; their locations are set in
`deploy/site.env`.

## Building

Build on a compute node, never on a login node, then copy the image to the
container folder on research storage under a new dated name, for example
`text_lab_DDMMYY.sif`. Do not move or delete existing images there: running
sessions may still use them. Test the new image in a sandbox app first by
setting `TL_CONTAINER` in `template/dev.env` (see the developer guide), then
switch production to it.

To add a dependency, edit the `%post` section of `text_lab.def` and rebuild.
Keep the version pins: the image combines several GPU stacks (WhisperX,
vLLM, PaddlePaddle, Ollama) whose versions constrain each other.
