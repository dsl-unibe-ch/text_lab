"""Historic handwriting: line-by-line transcription of manuscripts.

Pages are split into text lines by kraken's segmenter, and each line is read
by a handwriting recognition model. Interfaces use :mod:`.service`; see
``README.md`` in this folder for the pipeline.

The line-detection worker imports this package in another environment of
the image, so this file must not import anything.
"""
