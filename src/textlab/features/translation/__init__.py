"""Machine translation of texts and documents with local models.

Interfaces use :mod:`.service`. The other modules, by layer:

* Service: :mod:`.service`, the API for interfaces; :mod:`.messages`,
  error messages for users.
* Documents (:mod:`.documents`): Markdown and plain text, Office files,
  PDF reconstruction, and PDF to Markdown through the OCR feature.
* Protection: :mod:`.shield` keeps links, code, math, placeholders and
  glossary terms out of the model's reach.
* Backends: :mod:`.engine` (backend table, loading, dispatch),
  :mod:`.hf_backend` (Hugging Face batching and recovery),
  :mod:`.ollama_backend` (prompted translation with context budgets),
  :mod:`.chunking` (lossless splitting and limit errors).
* Resources: :mod:`.gpu_memory` (the lock serializing model use, CUDA
  cleanup) and :mod:`.gpu_profile` (batch sizes for the allocated GPU).
* Languages: :mod:`.lang_detect` detects the source language.
* Review: :mod:`.review` builds side-by-side review files.

See ``README.md`` in this folder for the pipeline and the files written.
"""
