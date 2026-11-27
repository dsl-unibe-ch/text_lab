"""Finding topics in a collection of texts: the Topic Modeling feature's API.

Interfaces call this module. :func:`load_table` and :func:`load_archive`
read an upload, and :func:`summarize_table` describes it for choosing its
columns; :func:`run_topic_modeling` analyzes it in a worker process
(:mod:`.worker`), so all GPU memory is released when the run ends and the
modeling libraries are never imported by the app. :func:`analyze_table` is
the analysis itself, in the calling process, for the worker and for batch
jobs. :func:`save_result` writes a result's downloads to a folder an
interface serves them from, and returns what it needs to show the result.

This module and the names it re-exports import only pandas; the modeling
modules are imported inside :func:`analyze_table`.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from textlab.common.jobs import WorkerError, run_worker
from textlab.common.outputs import OutputFolder, SavedFile
from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.common.storage import get_workspace
from textlab.features.topic_modeling.data import (
    TableSummary,
    load_archive,
    load_table,
    summarize_table,
)
from textlab.features.topic_modeling.models import (
    DEFAULT_EMBEDDING_MODEL_ENGLISH,
    DEFAULT_EMBEDDING_MODEL_MULTILINGUAL,
    DEFAULT_TIME_BINS,
    STABILITY_RUNS,
    SUPPORTED_LANGUAGES,
    TABLE_EXTENSIONS,
    TOP2VEC_BACKEND_LABELS,
    Algorithm,
    Notice,
    TopicModelingConfig,
    TopicModelingResult,
    TopicOverview,
)

__all__ = [
    "analyze_table",
    "load_archive",
    "load_table",
    "run_topic_modeling",
    "save_result",
    "summarize_table",
    # Results for interfaces
    "SavedFile",
    "SavedTopics",
    "TableSummary",
    "TopicOverview",
    # Re-exported for interfaces
    "DEFAULT_EMBEDDING_MODEL_ENGLISH",
    "DEFAULT_EMBEDDING_MODEL_MULTILINGUAL",
    "DEFAULT_TIME_BINS",
    "STABILITY_RUNS",
    "SUPPORTED_LANGUAGES",
    "TABLE_EXTENSIONS",
    "TOP2VEC_BACKEND_LABELS",
    "Algorithm",
    "Notice",
    "TopicModelingConfig",
    "TopicModelingResult",
]

LOGGER = logging.getLogger(__name__)

WORKER_MODULE = "textlab.features.topic_modeling.worker"

#: Workspace area for run folders and worker jobs.
WORKSPACE_AREA = "topic_modeling"


def run_topic_modeling(
    table: pd.DataFrame,
    config: TopicModelingConfig,
    *,
    source_name: str,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> TopicModelingResult:
    """Find the topics of a collection in a worker process.

    Args:
        table: The collection, one document per row (see
            :func:`load_table` and :func:`load_archive`).
        config: The run's settings, including the text column.
        source_name: The uploaded file's name, for the report.
        on_progress: Receives progress updates.
        cancel: Stops the worker when set.

    Returns:
        The topics, charts, metrics, download and notices.

    Raises:
        ValueError: If the run failed for a reason users can act on, such
            as a collection too small to cluster; the message says what to
            do.
        textlab.common.jobs.WorkerError: If the run failed otherwise; its
            ``details`` hold the worker's traceback.
        textlab.common.progress.CancelledError: If ``cancel`` was set.
    """
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="run-") as run_dir:
        table_path = run_dir / "table.pkl"
        zip_path = run_dir / "results.zip"
        # A pickle keeps every column's type; only this run's worker reads it.
        table.to_pickle(table_path)
        request = {
            "table": str(table_path),
            "zip": str(zip_path),
            "source_name": source_name,
            "config": config.to_dict(),
        }
        on_progress(Progress("Starting topic modeling worker..."))
        try:
            data = run_worker(
                WORKER_MODULE,
                request,
                area=WORKSPACE_AREA,
                on_progress=on_progress,
                cancel=cancel,
            )
        except WorkerError as error:
            if error.error_type == "ValueError":
                raise ValueError(error.error_message) from error
            raise
        return TopicModelingResult.from_dict(data, zip_path.read_bytes())


def analyze_table(
    table: pd.DataFrame,
    config: TopicModelingConfig,
    *,
    source_name: str,
    on_progress: ProgressCallback = no_progress,
) -> TopicModelingResult:
    """Find the topics of a collection in this process.

    Rows without text are dropped; for BERTopic with topics over time, so
    are rows whose timestamp cannot be read. BERTopic embeddings are
    computed once and reused by the stability runs. Use
    :func:`run_topic_modeling` from an interactive app.

    Args:
        table: The collection, one document per row.
        config: The run's settings.
        source_name: The uploaded file's name, for the report.
        on_progress: Receives progress updates.

    Returns:
        The topics, charts, metrics, download and notices.

    Raises:
        ValueError: If the collection cannot be analyzed with these
            settings, with a message for users.
    """
    from textlab.features.topic_modeling import (
        embeddings,
        evaluation,
        pipeline,
        reports,
    )
    from textlab.features.topic_modeling.data import (
        drop_empty_text_rows,
        prepare_timestamps,
    )
    from textlab.features.topic_modeling.overview import build_overview

    notices = []
    prepared = drop_empty_text_rows(table, config.text_column)

    timestamps = None
    if config.date_column and config.algorithm == Algorithm.BERTOPIC:
        prepared, timestamps, dropped = prepare_timestamps(
            prepared, config.date_column
        )
        if dropped:
            notices.append(
                Notice(
                    "warning",
                    f"{dropped} rows were skipped because "
                    f"'{config.date_column}' could not be parsed as a "
                    "date/time.",
                )
            )

    texts = prepared[config.text_column].astype(str).tolist()

    embedding_model = None
    vectors = None
    if config.algorithm == Algorithm.BERTOPIC:
        model_id = embeddings.resolve_bertopic_embedding_model_id(config)
        on_progress(Progress(f"Loading embedding model '{model_id}'..."))
        embedding_model, model_id = embeddings.load_embedding_model(config)
        notice = embeddings.long_document_notice(
            embedding_model, model_id, texts, config.chunk_long_documents
        )
        if notice is not None:
            notices.append(notice)
        on_progress(Progress("Encoding documents with the embedding model..."))
        vectors = embeddings.embed_documents(
            embedding_model,
            texts,
            chunk_long_documents=config.chunk_long_documents,
        )

    on_progress(
        Progress(
            f"Running topic extraction (Run 1/{STABILITY_RUNS})..."
            if config.evaluate_stability
            else "Running topic extraction..."
        )
    )
    run = pipeline.run_topic_modeling_pipeline(
        prepared,
        config,
        timestamps=timestamps,
        embedding_model=embedding_model,
        precomputed_embeddings=vectors,
    )

    on_progress(Progress("Calculating Topic Coherence and Diversity..."))
    metrics = evaluation.evaluate_run(run, texts, config)

    if config.evaluate_stability:
        on_progress(
            Progress(
                f"Running stability iterations 2-{STABILITY_RUNS} (unlocked "
                "seeds) and comparing topics..."
            )
        )
        metrics["Topic Stability"] = pipeline.evaluate_topic_stability(
            prepared,
            config,
            run["topic_keywords"],
            embedding_model=embedding_model,
            precomputed_embeddings=vectors,
        )

    overview = None
    if "topics" in run:
        try:
            overview = build_overview(
                run["topics"],
                run["docs_df"],
                config.text_column,
                topic_vectors=run.get("topic_vectors"),
                timestamps=timestamps,
                time_bins=config.time_bins,
            )
        except Exception:  # the charts are extra; keep the run's results
            LOGGER.exception("Could not build the topic overview.")
            notices.append(
                Notice(
                    "warning",
                    "The charts of this run could not be drawn; the topic "
                    "table and the download are complete.",
                )
            )

    report = reports.generate_metadata_report(
        filename=source_name,
        config=config,
        embedding_model_name=embeddings.get_embedding_model_name(config),
        evaluation_metrics=metrics,
    )
    zip_bytes = reports.build_results_zip(
        metadata_report=report,
        docs_df=run["docs_df"],
        topic_df=run["topic_df"],
        dashboard_assets=run["dashboard_assets"],
    )
    return TopicModelingResult(
        algorithm=config.algorithm,
        enable_dtm=config.enable_dtm,
        topic_df=run["topic_df"],
        dashboard_assets=run["dashboard_assets"],
        evaluation_metrics=metrics,
        zip_bytes=zip_bytes,
        notices=notices,
        overview=overview,
    )


# ---------------------------------------------------------------------------
# Results for interfaces
# ---------------------------------------------------------------------------

#: The files of the result ZIP offered on their own: name in the ZIP, the
#: suffix of the download, and what it is.
ZIP_DOWNLOADS = (
    (
        "document_topics.csv",
        "_document_topics.csv",
        "Documents with their topic, CSV",
        "text/csv; charset=utf-8",
    ),
    (
        "topic_keywords.csv",
        "_topic_keywords.csv",
        "Topics and keywords, CSV",
        "text/csv; charset=utf-8",
    ),
    (
        "run_configuration.txt",
        "_run_report.txt",
        "Settings and scores",
        "text/plain; charset=utf-8",
    ),
)


@dataclasses.dataclass(frozen=True)
class SavedTopics:
    """What :func:`save_result` wrote, and what to show of a run.

    The result without its ZIP and HTML charts, which are in the files.

    Attributes:
        algorithm: The algorithm that ran.
        enable_dtm: Whether topics over time were analyzed.
        columns: The topic table's columns.
        rows: The topic table's rows.
        metrics: Metric values by name; ``None`` when one could not be
            computed.
        notices: What users should know about the run.
        overview: The topics, their map and timeline, to draw, or
            ``None``.
        files: Every file written, downloads first.
    """

    algorithm: Algorithm
    enable_dtm: bool
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    metrics: dict[str, float | None]
    notices: tuple[Notice, ...]
    overview: TopicOverview | None
    files: tuple[SavedFile, ...]


def save_result(
    result: TopicModelingResult, source_name: str, folder: Path
) -> SavedTopics:
    """Write a result's downloads into a folder.

    The downloads are everything in one ZIP (with the interactive charts),
    and, from it, the documents with their topic, the topics with their
    keywords and the run's settings and scores.

    Args:
        result: The result of :func:`run_topic_modeling`.
        source_name: The analyzed file's name, which names the downloads.
        folder: An existing folder, such as a job's outputs.

    Returns:
        What was written, and what to show.
    """
    stem = PurePosixPath(source_name).stem or "collection"
    output = OutputFolder(folder)
    output.write(
        f"{stem}_topics.zip",
        result.zip_bytes,
        "Everything, with the interactive charts",
        "application/zip",
    )
    with zipfile.ZipFile(io.BytesIO(result.zip_bytes)) as archive:
        names = set(archive.namelist())
        for member, suffix, label, media_type in ZIP_DOWNLOADS:
            if member in names:
                output.write(
                    f"{stem}{suffix}", archive.read(member), label, media_type
                )
    table = result.topic_df
    return SavedTopics(
        algorithm=result.algorithm,
        enable_dtm=result.enable_dtm,
        columns=tuple(str(column) for column in table.columns),
        rows=tuple(
            tuple(_plain(value) for value in row)
            for row in table.itertuples(index=False)
        ),
        metrics=dict(result.evaluation_metrics),
        notices=tuple(result.notices),
        overview=result.overview,
        files=tuple(output.files),
    )


def _plain(value: Any) -> Any:
    """Return a table cell as a plain value; missing values as ``""``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return value.item() if hasattr(value, "item") else value
