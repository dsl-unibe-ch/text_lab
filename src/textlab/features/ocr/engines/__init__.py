"""Manual OCR engines: EasyOCR, PaddleOCR, OlmOCR and GLM-OCR.

Each engine is an adapter with the interface of :class:`.base.Engine`;
:mod:`textlab.features.ocr.manual` runs them on documents and batches.

This package is imported by the PaddleOCR worker, which runs in another
environment, so this file must not import anything.
"""
