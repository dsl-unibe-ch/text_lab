"""The Topic Modeling page: the themes that run through a set of texts.

The collection is a table with a text column (CSV, Excel) or a ZIP of text
files. Choosing it uploads it at once: it is read and described (its size,
columns and first rows), so the user can choose the text column and, for
topics over time, the timestamp column. The upload waits in a draft job
(``JobRunner.draft``) while the user chooses the algorithm (BERTopic,
Top2Vec, LDA) and its settings; every run copies it into a job of its own,
so the same collection can be run again with other settings. The analysis
runs in the feature's worker (``features.topic_modeling.service``).

The result shows the topics beside a map where alike topics lie close,
how alike each two topics are, how often each occurs over time, and the
quality scores, all drawn in SVG from the run's numbers (:class:`TopicsView`);
the libraries' interactive charts are in the download.
"""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
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

#: The quality scores: how each reads, whether it is a share, and what it
#: says.
METRICS = {
    "Topic Diversity": (
        True,
        "Share of distinct words among the topics' top ten keywords. "
        "Higher: the topics overlap less.",
    ),
    "Coherence (C_v)": (
        False,
        "How often a topic's keywords occur together in the texts, from 0 "
        "to 1. Higher reads better.",
    ),
    "Coherence (C_npmi)": (
        False,
        "Normalized pointwise mutual information of the keywords, from -1 "
        "to 1. Higher is better.",
    ),
    "Coherence (U_mass)": (
        False,
        "Co-occurrence of the keywords within documents; usually negative. "
        "Closer to 0 is better.",
    ),
    "LDA Perplexity": (
        False,
        "How well the model predicts the documents it learned from. Lower "
        "is better; compare LDA runs on the same data only.",
    ),
    "Topic Stability": (
        True,
        f"Average overlap of matching topics across {service.STABILITY_RUNS} "
        "runs with other random seeds. 100% gives the same topics every "
        "time.",
    ),
}


#: How the quality scores are computed, in Markdown.
METRICS_EXPLANATION = """\
**Topic diversity** is the share of distinct words among the top ten
keywords of all topics, computed from the topics alone.

**Coherence** (C_v, C_npmi, U_mass) is computed with Gensim's
`CoherenceModel`, from how often the keywords occur together in the texts.
For LDA, the texts are read as the model read them (lemmas); for BERTopic
and Top2Vec as words in lower case, without lemmas, as their keywords are.
Keywords that do not occur that way (such as phrases) are skipped, and
topics with fewer than two left do not count. The scores depend on how the
texts are read, so compare runs on the same collection and language only.

**LDA perplexity** measures how well the model predicts the documents it
learned from, not new ones.

**Topic stability** is the average Jaccard similarity of the best-matching
topics across the runs with other random seeds. BERTopic's embeddings are
computed once and reused, so only the reduction and grouping vary.
"""


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
            copy_into, draft.inputs / collection.file, job.inputs
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


def copy_into(source: Path, folder: Path) -> Path:
    """Put a file into a folder: linked if it can be, else copied.

    Args:
        source: The file.
        folder: The folder.

    Returns:
        The file in the folder.
    """
    target = folder / source.name
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return target


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


# ---------------------------------------------------------------------------
# The result, drawn
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TopicView:
    """One topic, ready to draw.

    Attributes:
        topic: Its number.
        region: Its id on the page, which links its dot on the map to it.
        title: Its first keywords, as a name.
        documents: How many documents have it as their main topic.
        share: Their share of all documents, from 0 to 1.
        bar: The length of its size bar, from 0 to 100 (the largest
            topic is 100).
        keywords: Its keywords with the length of their weight bars, from
            0 to 100.
        examples: The beginning of a few of its documents.
        x: Its place on the map, from 0 to 100.
        y: Its place on the map, from 0 to 100.
        r: Its dot's radius on the map.
    """

    topic: int | str
    region: str
    title: str
    documents: int
    share: float
    bar: float
    keywords: tuple[tuple[str, float], ...]
    examples: tuple[str, ...]
    x: float = 50.0
    y: float = 50.0
    r: float = 3.0


@dataclasses.dataclass(frozen=True)
class Heatmap:
    """How alike each two topics are, ready to draw as a grid.

    Attributes:
        size: The width and height of the drawing, in its units.
        labels: Each topic's number and the centre of its row and column.
        cells: Each cell's left and top, its value from 0 to 1, and its
            title.
    """

    size: float
    labels: tuple[tuple[float, int | str], ...]
    cells: tuple[tuple[float, float, float, str], ...]


@dataclasses.dataclass(frozen=True)
class Timeline:
    """One topic over time, ready to draw in a 100 by 30 box.

    Attributes:
        topic: The topic.
        line: The line's points.
        area: The points of the area under the line.
        peak: The most documents in one interval.
        peak_at: That interval's label.
    """

    topic: TopicView
    line: str
    area: str
    peak: int
    peak_at: str


@dataclasses.dataclass(frozen=True)
class TopicsView:
    """A run's result, as the page draws it.

    Attributes:
        saved: What the service saved and returned.
        config: The settings it ran with.
    """

    saved: service.SavedTopics
    config: service.TopicModelingConfig

    @property
    def overview(self) -> service.TopicOverview | None:
        """The topics, map and timeline, if the run drew them."""
        return self.saved.overview

    @property
    def algorithm(self) -> str:
        """The algorithm's name."""
        return self.saved.algorithm.label

    @property
    def mixed(self) -> bool:
        """Whether each document is a mix of topics (LDA)."""
        return self.saved.algorithm == service.Algorithm.LDA

    def topics(self, job_id: str) -> list[TopicView]:
        """Return the topics, ready to draw.

        Args:
            job_id: The job's id, which makes the topics' ids unique.

        Returns:
            The topics, in the topic table's order.
        """
        overview = self.overview
        if overview is None:
            return []
        total = max(overview.documents, 1)
        largest = max((item.documents for item in overview.topics), default=1)
        positions = overview.positions or []
        views = []
        for index, item in enumerate(overview.topics):
            share = item.documents / total
            place: dict[str, float] = {}
            if index < len(positions):
                x, y = positions[index]
                place = {
                    "x": round(8 + 84 * x, 2),
                    "y": round(8 + 84 * (1 - y), 2),
                    "r": round(2 + 10 * math.sqrt(share), 2),
                }
            weights = item.weights or [1.0] * len(item.keywords)
            views.append(
                TopicView(
                    topic=item.topic,
                    region=f"{job_id}-topic-{index}",
                    title=" · ".join(item.keywords[:3]),
                    documents=item.documents,
                    share=share,
                    bar=round(100 * item.documents / max(largest, 1), 2),
                    keywords=tuple(
                        (word, round(100 * weight, 2))
                        for word, weight in zip(
                            item.keywords, weights, strict=False
                        )
                    ),
                    examples=tuple(item.examples),
                    **place,
                )
            )
        return views

    def heatmap(self, job_id: str) -> Heatmap | None:
        """Return the similarity heatmap, ready to draw.

        Args:
            job_id: The job's id.

        Returns:
            The heatmap, or ``None`` without similarities.
        """
        matrix = self.overview.similarity if self.overview else None
        if not matrix:
            return None
        names = [topic.topic for topic in self.topics(job_id)]
        cell, margin = 10, 14
        return Heatmap(
            size=margin + cell * len(names),
            labels=tuple(
                (margin + cell * index + cell / 2, name)
                for index, name in enumerate(names)
            ),
            cells=tuple(
                (
                    margin + cell * column,
                    margin + cell * row,
                    value,
                    f"Topics {names[row]} and {names[column]}: {value:.2f}",
                )
                for row, values in enumerate(matrix)
                for column, value in enumerate(values)
            ),
        )

    def timelines(self, job_id: str) -> list[Timeline]:
        """Return each topic's timeline, scaled to its own peak.

        Args:
            job_id: The job's id.

        Returns:
            One timeline per topic, in the topic table's order.
        """
        overview = self.overview
        if overview is None or not overview.over_time:
            return []
        labels = overview.time_labels
        lines = []
        for topic, counts in zip(
            self.topics(job_id), overview.over_time, strict=False
        ):
            peak = max(counts, default=0)
            step = 100 / (len(counts) - 1) if len(counts) > 1 else 0
            points = [
                (
                    round(index * step if step else 50, 2),
                    round(28 - 26 * count / peak, 2) if peak else 28,
                )
                for index, count in enumerate(counts)
            ]
            line = " ".join(f"{x},{y}" for x, y in points)
            lines.append(
                Timeline(
                    topic=topic,
                    line=line,
                    area=f"{points[0][0]},30 {line} {points[-1][0]},30",
                    peak=peak,
                    peak_at=labels[counts.index(peak)] if peak else "",
                )
            )
        return lines

    @property
    def explanation(self) -> str:
        """How the quality scores are computed, in Markdown."""
        return METRICS_EXPLANATION

    @property
    def metrics(self) -> list[tuple[str, str, str]]:
        """The quality scores: name, value for people, and what it says."""
        found = []
        for name, value in self.saved.metrics.items():
            share, meaning = METRICS.get(name, (False, ""))
            if value is None:
                shown = "Not computed"
            elif share:
                shown = f"{value:.0%}"
            else:
                shown = f"{value:.3f}"
            found.append((name, shown, meaning))
        return found
