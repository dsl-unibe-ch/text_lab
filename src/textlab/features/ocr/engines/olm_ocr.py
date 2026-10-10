"""OlmOCR: PDFs to clean Markdown, tuned for scientific papers.

OlmOCR needs its own environment (``common.container.OLMOCR_ENV``) and runs
its own pipeline (``python -m olmocr.pipeline``) with vLLM on the GPU. It
reads whole PDFs, so an image is converted to a one-page PDF first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from textlab.common import container, jobs
from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.features.ocr.engines.base import (
    Engine,
    EngineError,
    EngineOptions,
    EngineOutput,
    PageText,
    process_output,
    run_process,
)

#: Share of GPU memory vLLM may take.
GPU_MEMORY_UTILIZATION = os.environ.get("OLMOCR_GPU_MEMORY_UTILIZATION", "0.6")


class OlmOcr(Engine):
    """OlmOCR's pipeline on one PDF."""

    name = "OlmOCR"

    def recognize(
        self,
        input_path: Path,
        work_dir: Path,
        options: EngineOptions,
        on_progress: ProgressCallback = no_progress,
    ) -> EngineOutput:
        """Read one PDF or image (see :meth:`Engine.recognize`).

        The whole document is returned as one page, with OlmOCR's own
        result record.
        """
        on_progress(Progress(f"Running {self.name}..."))
        work_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = input_path
        if input_path.suffix.lower() != ".pdf":
            pdf_path = _image_to_pdf(input_path, work_dir)

        python = container.env_python(
            container.OLMOCR_ENV, "OLMOCR_BACKEND_PYTHON"
        )
        olmocr_dir = work_dir / "olmocr"
        command = [
            python,
            "-m",
            "olmocr.pipeline",
            str(olmocr_dir),
            "--markdown",
            "--pdfs",
            str(pdf_path),
            "--gpu-memory-utilization",
            GPU_MEMORY_UTILIZATION,
        ]
        result = run_process(command, jobs.worker_environment(python))
        if result.returncode != 0:
            raise EngineError(
                f"OCR process failed. Code: {result.returncode}",
                details=process_output(result),
            )
        records = sorted((olmocr_dir / "results").glob("*.jsonl"))
        if not records:
            raise EngineError(
                "No .jsonl output found.", details=process_output(result)
            )
        with records[0].open(encoding="utf-8") as file:
            record = file.readline().strip()
        data = json.loads(record)
        return EngineOutput(
            pages=[PageText(1, data.get("text") or "", data)], record=record
        )


def _image_to_pdf(image_path: Path, work_dir: Path) -> Path:
    """Save an image as a one-page PDF in ``work_dir``."""
    from PIL import Image

    pdf_path = work_dir / f"{image_path.stem}.pdf"
    try:
        with Image.open(image_path) as image:
            image.convert("RGB").save(pdf_path, "PDF", resolution=100.0)
    except Exception as error:
        raise EngineError(
            f"Failed to convert image to PDF for OlmOCR: {error}"
        ) from error
    return pdf_path
