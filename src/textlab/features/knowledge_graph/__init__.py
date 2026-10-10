"""Backend for the Knowledge Graph feature: graphs of a paper collection.

Interfaces use :mod:`.service`. The modules behind it:

- ``grobid``: the Grobid server that parses PDFs into TEI XML.
- ``tei``: metadata, references and plain text from TEI XML.
- ``corpus``: the corpus folder: parsing new papers, the corpus tables.
- ``topics``: topics per paper from an LLM (Ollama or GPUStack).
- ``graphs``: ego and full corpus graphs (NetworkX, Pyvis).
- ``models``: the results of the steps.

See ``README.md`` in this folder for the pipeline and the files written.
"""
