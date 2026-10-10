"""Worker process for topic modeling; started by ``run_topic_modeling``.

Runs :func:`textlab.features.topic_modeling.service.analyze_table` in its
own process, so the modeling libraries stay out of the app and all GPU
memory the embedding model used is released when it exits::

    python -m textlab.features.topic_modeling.worker JOB_DIR

``JOB_DIR`` holds the request; see :mod:`textlab.common.jobs` for the
protocol. Not meant to be run by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from textlab.common.jobs import worker_main
from textlab.common.progress import ProgressCallback
from textlab.features.topic_modeling.models import TopicModelingConfig
from textlab.features.topic_modeling.service import analyze_table


def handle(request: Mapping[str, Any], report: ProgressCallback) -> dict:
    """Analyze the collection named in a request.

    Args:
        request: ``{"table", "zip", "source_name", "config"}``: the pickled
            collection, where to write the result ZIP, the uploaded file's
            name and the configuration.
        report: Receives progress updates.

    Returns:
        The result (``models.TopicModelingResult``) as a dictionary,
        without the ZIP.
    """
    table = pd.read_pickle(request["table"])
    config = TopicModelingConfig.from_dict(request["config"])
    result = analyze_table(
        table, config, source_name=request["source_name"], on_progress=report
    )
    Path(request["zip"]).write_bytes(result.zip_bytes)
    return result.to_dict()


if __name__ == "__main__":
    worker_main(handle)
