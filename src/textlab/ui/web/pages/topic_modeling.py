"""The Topic Modeling page: the themes that run through a set of texts.

The collection is a table with a text column (CSV, Excel) or a ZIP of text
files. Choosing it uploads it at once: it is read and described (its size,
columns and first rows), so the user can choose the text column and, for
topics over time, the timestamp column. The upload waits in a draft job
(``JobRunner.draft``) while the user chooses the algorithm (BERTopic,
Top2Vec, LDA) and its settings; every run copies it into a job of its own,
so the same collection can be run again with other settings. The analysis
runs in the feature's worker (``features.topic_modeling.service``).

The result is drawn as each algorithm's own library draws it: LDA as
pyLDAvis shows it, BERTopic and Top2Vec with their usual charts, all in
SVG from the run's numbers (:mod:`.topic_charts`); the libraries'
interactive charts are in the download.
"""

from __future__ import annotations

import dataclasses
import math
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData

from textlab.common import gpu_manager
from textlab.common.progress import Progress
from textlab.features.topic_modeling import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import Download, JobContext, render_card, runner_of
from textlab.ui.web.pages.topic_charts import TopicsView
from textlab.ui.web.render import renderer

router = protected_router()

#: The page's id, which also names its jobs.
PAGE = "topic_modeling"

#: What the page reads: tables, or a ZIP of text files.
TABLE_EXTENSIONS = tuple(f".{ext}" for ext in service.TABLE_EXTENSIONS)
EXTENSIONS = (*TABLE_EXTENSIONS, ".zip")

#: The text column of a ZIP's table (``service.load_archive``).
ARCHIVE_TEXT_COLUMN = "Text"

#: The result's template.
RESULT_TEMPLATE = "topic_modeling/result.html"

#: The embedding model choices for BERTopic: form value and model id.
EMBEDDING_MODELS = {
    "default": None,
    "english": service.DEFAULT_EMBEDDING_MODEL_ENGLISH,
    "multilingual": service.DEFAULT_EMBEDDING_MODEL_MULTILINGUAL,
}

#: Dimension reduction choices for BERTopic: form value, the setting's
#: value, and the choice's name.
REDUCTIONS = {
    "umap": ("UMAP", "UMAP, keeps local and global structure"),
    "pca": ("PCA", "PCA, fast and linear"),
    "svd": ("Truncated SVD", "Truncated SVD, fast and linear"),
    "none": ("None", "None, cluster the embeddings as they are"),
}

#: Top2Vec's training depths.
TRAINING_DEPTHS = ("fast-learn", "learn", "deep-learn")


@dataclasses.dataclass(frozen=True)
class Collection:
    """An uploaded collection waiting in a draft job.

    Attributes:
        name: Its file name, as uploaded.
        file: Its file in the draft's inputs.
        archive: Whether it is a ZIP of text files.
        summary: Its size, columns and first rows.
    """

    name: str
    file: str
    archive: bool
    summary: service.TableSummary


def collections(request: Request) -> dict[str, Collection]:
    """Return the session's uploaded collections, by draft job id.

    Args:
        request: The current request.

    Returns:
        The app's record of them; empty at first.
    """
    state = request.app.state
    found = getattr(state, "topic_collections", None)
    if found is None:
        found = state.topic_collections = {}
    return found


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/topic-modeling")
def page(request: Request) -> HTMLResponse:
    """Show the form and the session's topic modeling jobs.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    jobs = runner_of(request).jobs(PAGE)
    return renderer(request).response(
        request,
        "topic_modeling/page.html",
        page_id=PAGE,
        accept=",".join(EXTENSIONS),
        formats=uploads.format_names(TABLE_EXTENSIONS) + ", or a ZIP of TXT",
        algorithms=list(service.Algorithm),
        languages=service.SUPPORTED_LANGUAGES,
        embedding_english=service.DEFAULT_EMBEDDING_MODEL_ENGLISH,
        embedding_multilingual=service.DEFAULT_EMBEDDING_MODEL_MULTILINGUAL,
        reductions=REDUCTIONS,
        depths=TRAINING_DEPTHS,
        top2vec_backends=service.TOP2VEC_BACKEND_LABELS,
        stability_runs=service.STABILITY_RUNS,
        cards=Markup("".join(render_card(request, job) for job in jobs[::-1])),
    )


@router.post("/topic-modeling/collection")
async def upload_collection(
    request: Request, session: Checked
) -> HTMLResponse:
    """Read an uploaded collection and show its columns to choose from.

    The upload is kept in a draft job until the session ends or another
    collection replaces it (``collection``, the previous one's id).

    Args:
        request: The form: ``file`` and, after a first upload,
            ``collection``.
        session: The session, with its CSRF token checked.

    Returns:
        The collection's description, with the column choices.

    Raises:
        UploadError: If the file is missing, of another type, or holds no
            documents.
    """
    form = await uploads.read_form(request, max_files=1, max_fields=2)
    runner = runner_of(request)
    known = collections(request)
    previous = str(form.get("collection") or "")
    if previous in known:
        draft = runner.draft(previous)
        known.pop(previous)
        if draft is not None:
            runner.discard(draft)
    upload = uploads.single_file(form, "file", suffixes=EXTENSIONS)
    name = uploads.display_name(upload)
    archive = PurePosixPath(name).suffix.lower() == ".zip"
    draft = runner.create(title=name, page=PAGE)
    try:
        path = await run_in_threadpool(uploads.save, upload, draft.inputs)
        table = await run_in_threadpool(load, path, name, archive)
        summary = await run_in_threadpool(service.summarize_table, table)
    except ValueError as error:
        runner.discard(draft)
        raise uploads.UploadError(str(error)) from error
    except BaseException:
        runner.discard(draft)
        raise
    known[draft.id] = Collection(name, path.name, archive, summary)
    return renderer(request).response(
        request,
        "topic_modeling/collection.html",
        collection_id=draft.id,
        collection=known[draft.id],
        time_bins=service.DEFAULT_TIME_BINS,
    )


@router.post("/topic-modeling")
async def find_topics(request: Request, session: Checked) -> HTMLResponse:
    """Start a run on the uploaded collection with the chosen settings.

    Args:
        request: The form: ``collection``, the columns and the settings.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        UploadError: If there is no collection or a setting is not valid.
    """
    form = await uploads.read_form(request, max_files=0, max_fields=80)
    runner = runner_of(request)
    key = str(form.get("collection") or "")
    collection = collections(request).get(key)
    draft = runner.draft(key) if collection else None
    if collection is None or draft is None:
        raise uploads.UploadError(
            "Choose the collection first: a table or a ZIP of text files."
        )
    config = read_config(form, collection)
    job = runner.create(
        title=collection.name,
        page=PAGE,
        gpu=gpu_manager.TOPIC_MODELING,
        result_template=RESULT_TEMPLATE,
    )
    try:
        path = await run_in_threadpool(
            uploads.copy_into, draft.inputs / collection.file, job.inputs
        )
    except BaseException:
        runner.discard(job)
        raise
    runner.submit(
        job,
        partial(analyze, path, collection.name, collection.archive, config),
    )
    return HTMLResponse(render_card(request, job))


def load(path: Path, name: str, archive: bool) -> Any:
    """Read a collection from its file.

    Args:
        path: The file.
        name: Its name as uploaded, which picks a table's reader.
        archive: Whether it is a ZIP of text files.

    Returns:
        The collection, one document per row.

    Raises:
        ValueError: If it holds no documents or cannot be read.
    """
    data = path.read_bytes()
    if archive:
        return service.load_archive(data)
    return service.load_table(name, data)


# ---------------------------------------------------------------------------
# The settings
# ---------------------------------------------------------------------------


def number(
    form: FormData,
    field: str,
    default: float,
    low: float,
    high: float,
    label: str,
    *,
    whole: bool = True,
) -> Any:
    """Read a number from the form, or its default when it is not sent.

    Args:
        form: The form.
        field: The field's name.
        default: The value when the field is missing or empty.
        low: The smallest value allowed.
        high: The largest value allowed.
        label: The setting's name, for the error message.
        whole: Whether the value must be a whole number.

    Returns:
        The number, as ``int`` when ``whole``.

    Raises:
        UploadError: If it is not a number in the range.
    """
    raw = str(form.get(field) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (low <= value <= high) or (whole and not value.is_integer()):
        kind = "a whole number" if whole else "a number"
        raise uploads.UploadError(
            f"{label} must be {kind} from {low:g} to {high:g}."
        )
    return int(value) if whole else value


def checked(form: FormData, field: str) -> bool:
    """Return whether a check box was checked."""
    return bool(form.get(field))


def choice(
    form: FormData, field: str, options: Any, default: str, label: str
) -> str:
    """Read one of a set of values from the form.

    Args:
        form: The form.
        field: The field's name.
        options: The values allowed.
        default: The value when the field is missing.
        label: The setting's name, for the error message.

    Returns:
        The value.

    Raises:
        UploadError: If it is not one of ``options``.
    """
    value = str(form.get(field) or default)
    if value not in options:
        raise uploads.UploadError(f"Choose the {label} from the list.")
    return value


def read_config(
    form: FormData, collection: Collection
) -> service.TopicModelingConfig:
    """Read the run's settings from the form.

    Args:
        form: The form.
        collection: The collection, whose columns the choices refer to.

    Returns:
        The configuration.

    Raises:
        UploadError: If a setting is not valid.
    """
    columns = collection.summary.columns
    if collection.archive:
        text_column = ARCHIVE_TEXT_COLUMN
    else:
        text_column = choice(form, "text_column", columns, "", "text column")
    algorithm = service.Algorithm(
        choice(
            form,
            "algorithm",
            [str(item) for item in service.Algorithm],
            str(service.Algorithm.BERTOPIC),
            "algorithm",
        )
    )
    date_column = None
    time_bins = service.DEFAULT_TIME_BINS
    over_time = (
        not collection.archive
        and algorithm == service.Algorithm.BERTOPIC
        and checked(form, "over_time")
    )
    if over_time:
        date_column = choice(
            form, "date_column", columns, "", "timestamp column"
        )
        if date_column == text_column:
            raise uploads.UploadError(
                "The timestamp column must be another than the text column."
            )
        time_bins = number(
            form, "time_bins", time_bins, 2, 100, "The number of intervals"
        )
    settings = {
        service.Algorithm.BERTOPIC: bertopic_settings,
        service.Algorithm.TOP2VEC: top2vec_settings,
        service.Algorithm.LDA: lda_settings,
    }[algorithm](form)
    return service.TopicModelingConfig(
        algorithm=algorithm,
        language=choice(
            form,
            "language",
            service.SUPPORTED_LANGUAGES,
            service.SUPPORTED_LANGUAGES[0],
            "language",
        ),
        text_column=text_column,
        enable_dtm=date_column is not None,
        date_column=date_column,
        time_bins=time_bins,
        evaluate_stability=checked(form, "stability"),
        **settings,
    )


def topic_count(form: FormData, prefix: str, high: int) -> int | str:
    """Read whether to find the number of topics, or which to use.

    Args:
        form: The form.
        prefix: The algorithm's field prefix.
        high: The most topics allowed.

    Returns:
        ``"auto"``, or the number.
    """
    if str(form.get(f"{prefix}_count") or "auto") == "auto":
        return "auto"
    return number(
        form, f"{prefix}_topics", 10, 2, high, "The number of topics"
    )


def bertopic_settings(form: FormData) -> dict[str, Any]:
    """Read BERTopic's settings.

    Args:
        form: The form.

    Returns:
        The configuration's BERTopic fields.

    Raises:
        UploadError: If a setting is not valid.
    """
    settings: dict[str, Any] = {
        "custom_stopwords": str(form.get("bt_stopwords") or "").strip(),
        "ngram_range": (1, 2) if checked(form, "bt_phrases") else (1, 1),
        "min_df": number(
            form, "bt_min_df", 1, 1, 500, "The minimum topic frequency"
        ),
        "reduce_frequent": checked(form, "bt_reduce_frequent"),
        "chunk_long_documents": checked(form, "bt_chunk"),
    }
    if (
        choice(
            form,
            "bt_clustering",
            ("hdbscan", "kmeans"),
            "hdbscan",
            "clustering",
        )
        == "kmeans"
    ):
        clusters = number(
            form, "bt_clusters", 10, 2, 100, "The number of topics"
        )
        settings |= {
            "clustering_algo": "KMeans",
            "clustering_params": {"n_clusters": clusters},
            "num_topics": None,
            "reduce_outliers": False,
        }
    else:
        size = number(
            form, "bt_min_size", 10, 3, 500, "The documents per topic"
        )
        settings |= {
            "clustering_algo": "HDBSCAN",
            "clustering_params": {
                "min_cluster_size": size,
                "min_samples": number(
                    form,
                    "bt_min_samples",
                    size,
                    1,
                    500,
                    "The outlier sensitivity",
                ),
            },
            "num_topics": topic_count(form, "bt", 100),
            "reduce_outliers": checked(form, "bt_reduce_outliers"),
        }

    embedding = choice(
        form,
        "bt_embedding",
        (*EMBEDDING_MODELS, "custom"),
        "default",
        "embedding model",
    )
    trust = False
    if embedding == "custom":
        model_id = str(form.get("bt_custom_model") or "").strip()
        if not model_id:
            raise uploads.UploadError(
                "Name the embedding model, or choose one of the others."
            )
        trust = checked(form, "bt_trust_code")
    else:
        model_id = EMBEDDING_MODELS[embedding]
    settings |= {"embedding_model_id": model_id, "trust_remote_code": trust}

    reduction = choice(
        form, "bt_reduction", REDUCTIONS, "umap", "dimension reduction"
    )
    params: dict[str, Any] = {}
    if reduction != "none":
        params["n_components"] = number(
            form, "bt_components", 5, 2, 50, "The target dimensions"
        )
        if reduction == "umap":
            params["n_neighbors"] = number(
                form, "bt_neighbors", 15, 2, 200, "The neighbors"
            )
            params["min_dist"] = number(
                form,
                "bt_min_dist",
                0.0,
                0.0,
                1.0,
                "The minimum distance",
                whole=False,
            )
        params["random_state"] = 42 if checked(form, "bt_lock_seed") else None
    settings |= {
        "dim_reduction_algo": REDUCTIONS[reduction][0],
        "dim_params": params,
    }
    return settings


def top2vec_settings(form: FormData) -> dict[str, Any]:
    """Read Top2Vec's settings.

    Args:
        form: The form.

    Returns:
        The configuration's Top2Vec fields.

    Raises:
        UploadError: If a setting is not valid.
    """
    return {
        "num_topics": topic_count(form, "t2v", 100),
        "top2vec_backend": choice(
            form,
            "t2v_backend",
            service.TOP2VEC_BACKEND_LABELS,
            "transformer",
            "embedding backend",
        ),
        "top2vec_speed": choice(
            form, "t2v_speed", TRAINING_DEPTHS, "learn", "training depth"
        ),
        "top2vec_min_count": number(
            form, "t2v_min_count", 10, 1, 500, "The minimum word count"
        ),
    }


def lda_settings(form: FormData) -> dict[str, Any]:
    """Read LDA's settings.

    Args:
        form: The form.

    Returns:
        The configuration's LDA fields.

    Raises:
        UploadError: If a setting is not valid.
    """
    return {
        "num_topics": number(
            form, "lda_topics", 10, 2, 50, "The number of topics"
        ),
        "custom_stopwords": str(form.get("lda_stopwords") or "").strip(),
        "use_bigrams": checked(form, "lda_bigrams"),
        "passes": number(form, "lda_passes", 10, 5, 50, "The passes"),
    }


# ---------------------------------------------------------------------------
# The job's work
# ---------------------------------------------------------------------------


def analyze(
    path: Path,
    name: str,
    archive: bool,
    config: service.TopicModelingConfig,
    context: JobContext,
) -> TopicsView:
    """Find the topics of a collection and offer the downloads.

    Args:
        path: The collection, in the job's inputs.
        name: Its name as uploaded.
        archive: Whether it is a ZIP of text files.
        config: The run's settings.
        context: The job's context.

    Returns:
        The result to show.
    """
    context.report(Progress("Reading the collection", 0.0))
    table = load(path, name, archive)
    result = service.run_topic_modeling(
        table,
        config,
        source_name=name,
        on_progress=context.report,
        cancel=context.cancel,
    )
    context.report(Progress("Saving the results", 1.0))
    saved = service.save_result(result, name, context.outputs)
    for item in saved.files:
        context.offer(
            Download(
                item.name,
                item.label,
                item.media_type,
                path=item.path,
                listed=item.listed,
            )
        )
    return TopicsView(saved=saved, config=config)
