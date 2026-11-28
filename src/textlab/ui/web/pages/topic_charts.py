"""The Topic Modeling result, drawn as each algorithm's library draws it.

:class:`TopicsView` wraps a saved run (``service.SavedTopics``) and works
out the geometry of its charts, in the units of each chart's SVG, so the
templates only place what they are given:

- **LDA**, as pyLDAvis shows it (:meth:`TopicsView.ldavis`): the topics on
  a map, sized by their share of the words, beside the terms ranked by
  relevance for the selected topic, with the lambda slider (``app.js``
  re-ranks them); and the documents with their mixture of topics.
- **BERTopic**, as its visualizations do: the keyword scores of each
  topic, the topic map, the documents placed by meaning
  (:meth:`TopicsView.document_plot`), the hierarchy of topics
  (:meth:`TopicsView.dendrogram`), the similarity heatmap in the
  hierarchy's order, and the topics over time with each interval's words
  (:meth:`TopicsView.trends`).
- **Top2Vec**: its word clouds, the topic sizes, the topic map, the
  documents placed by meaning and the heatmap.

Charts use emphasis rather than a colour per topic: everything in grey,
the topic pointed at or chosen in the accent colour; elements of one topic
share a ``data-region``, which ``app.js`` marks together.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from textlab.features.topic_modeling import service

#: The terms pyLDAvis lists for a topic.
LDAVIS_TERMS = 30

#: The lambda pyLDAvis starts with: terms ranked by probability alone.
LDAVIS_LAMBDA = 1.0

#: The largest radius of a topic's bubble, for a topic of all the words or
#: documents, in units of the 100 by 100 maps.
BUBBLE_RADIUS = 16.0

#: Shares shown in the size legend of the LDA map.
SIZE_LEGEND = (0.02, 0.05, 0.1)

#: The quality scores: whether each is a share, and what it says.
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


# ---------------------------------------------------------------------------
# The parts of the charts
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MixturePart:
    """A topic's part of a document, in its mixture bar (LDA).

    Attributes:
        topic: The topic's number, or "Other" for the rest.
        share: Its share of the document, from 0 to 1.
        x: Where the part starts, from 0 to 100.
        width: How wide it is.
        this: Whether it is the topic the document is listed under.
        shade: 0 or 1, alternating, to tell neighbouring parts apart.
    """

    topic: int | str
    share: float
    x: float
    width: float
    this: bool
    shade: int


@dataclasses.dataclass(frozen=True)
class TopicView:
    """One topic, ready to draw.

    Attributes:
        topic: Its number.
        index: Its place in the topic table, from 0.
        region: Its id on the page, which links its marks in every chart.
        title: Its first keywords, as a name.
        documents: How many documents have it as their main topic.
        share: Their share of all documents, from 0 to 1.
        prevalence: Its share of the collection as its library sizes it
            (of the words for LDA, else of the documents).
        bar: The length of its size bar, from 0 to 100 (the largest
            topic is 100).
        keywords: Its keywords with the length of their weight bars, from
            0 to 100.
        examples: The beginning of a few of its documents.
        mixtures: For each example, its mixture bar (LDA).
        cloud: Its word cloud: each word with its size, from 1 to 5
            (Top2Vec).
        x: Its place on the topic map, from 0 to 100.
        y: Its place on the topic map, from 0 to 100.
        r: Its bubble's radius on the topic map.
    """

    topic: int | str
    index: int
    region: str
    title: str
    documents: int
    share: float
    prevalence: float
    bar: float
    keywords: tuple[tuple[str, float], ...]
    examples: tuple[str, ...]
    mixtures: tuple[tuple[MixturePart, ...], ...] = ()
    cloud: tuple[tuple[str, int], ...] = ()
    x: float = 50.0
    y: float = 50.0
    r: float = 3.0


@dataclasses.dataclass(frozen=True)
class Heatmap:
    """How alike each two topics are, ready to draw as a grid.

    Attributes:
        size: The width and height of the drawing, in its units.
        labels: Each row's topic number and the centre of its row and
            column.
        cells: Each cell's left and top, its value from 0 to 1, and its
            title.
    """

    size: float
    labels: tuple[tuple[float, int | str], ...]
    cells: tuple[tuple[float, float, float, str], ...]

    @property
    def width(self) -> int:
        """Its width on the page, in pixels: 3 per unit, so 30 per topic."""
        return round(3 * self.size)


@dataclasses.dataclass(frozen=True)
class Dendrogram:
    """How the topics merge, ready to draw (a 600-wide drawing).

    Attributes:
        height: The drawing's height.
        leaves: Each topic in the leaves' order, with its row's middle.
        links: Each merge's outline, as SVG polyline points.
        ticks: The distance axis: each tick's x and label.
        axis_y: Where the distance axis lies.
        label_x: Where the leaves' labels end.
        left: Where the tree starts (distance 0).
        right: Where it ends (the last merge).
    """

    height: float
    leaves: tuple[tuple[TopicView, float], ...]
    links: tuple[str, ...]
    ticks: tuple[tuple[float, str], ...]
    axis_y: float
    label_x: float
    left: float
    right: float

    @property
    def tick_end(self) -> float:
        """Where the axis's ticks end, down."""
        return self.axis_y + 4

    @property
    def tick_label_y(self) -> float:
        """Where the axis's labels sit."""
        return self.axis_y + 16


@dataclasses.dataclass(frozen=True)
class DocumentPoint:
    """A document on the document map.

    Attributes:
        x: Its place, from 0 to 100.
        y: Its place, from 0 to 100.
        region: Its topic's region, or "" for no topic.
        excerpt: Its beginning.
    """

    x: float
    y: float
    region: str
    excerpt: str


@dataclasses.dataclass(frozen=True)
class DocumentPlot:
    """The documents placed by meaning, ready to draw (100 by 100).

    Attributes:
        points: The documents drawn.
        labels: Each topic's label at the middle of its documents: its
            number and place.
        drawn: How many documents are drawn.
        documents: How many there are in all.
    """

    points: tuple[DocumentPoint, ...]
    labels: tuple[tuple[TopicView, float, float], ...]
    drawn: int
    documents: int


@dataclasses.dataclass(frozen=True)
class TrendPoint:
    """One topic in one interval, on the over-time chart.

    Attributes:
        x: Its place across.
        y: Its place down.
        title: Its tooltip: the interval, the count and the words then.
    """

    x: float
    y: float
    title: str


@dataclasses.dataclass(frozen=True)
class Trend:
    """One topic's line on the over-time chart.

    Attributes:
        topic: The topic.
        line: The line's points.
        points: Its points, with their tooltips.
        peak: The most documents in one interval.
        peak_at: That interval's label.
        counts: Its documents per interval, for the table.
    """

    topic: TopicView
    line: str
    points: tuple[TrendPoint, ...]
    peak: int
    peak_at: str
    counts: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Trends:
    """The topics over time on one chart, ready to draw (600 by 260).

    Attributes:
        trends: One line per topic.
        labels: The intervals' labels.
        ticks_x: Labelled intervals: x and label.
        ticks_y: Count ticks: y and label.
        left: The plot's left edge.
        right: Its right edge.
        top: Its top edge.
        bottom: Its bottom edge (the base line).
    """

    trends: tuple[Trend, ...]
    labels: tuple[str, ...]
    ticks_x: tuple[tuple[float, str, str], ...]
    ticks_y: tuple[tuple[float, str], ...]
    left: float
    right: float
    top: float
    bottom: float

    @property
    def label_x(self) -> float:
        """Where the count labels end, across."""
        return self.left - 6

    @property
    def label_y(self) -> float:
        """Where the interval labels sit, down."""
        return self.bottom + 18


@dataclasses.dataclass(frozen=True)
class LdaTerm:
    """A term pyLDAvis lists, with what ranks and draws it.

    Attributes:
        term: The term.
        freq: Its estimated frequency in the topic (red bar); its overall
            frequency in the list of salient terms.
        total: Its overall frequency (grey bar).
        logprob: Its log probability in the topic.
        loglift: Its log lift: probability in the topic over overall.
        shares: How its occurrences spread over the topics, as "index:share"
            pairs, for the bubbles' sizes while it is pointed at.
        total_bar: The grey bar's length, from 0 to 100.
        freq_bar: The red bar's length.
        shown: Whether it is among the terms shown first (lambda 1).
    """

    term: str
    freq: float
    total: float
    logprob: float
    loglift: float
    shares: str
    total_bar: float
    freq_bar: float
    shown: bool


@dataclasses.dataclass(frozen=True)
class LdaTopic:
    """A topic in pyLDAvis's view: its bubble and its terms.

    Attributes:
        topic: The topic.
        x: Its bubble's place, from 0 to 100.
        y: Its bubble's place.
        r: Its bubble's radius.
        terms: The terms it may rank, the first at lambda 1 first.
        caption: What its list is, for people.
    """

    topic: TopicView
    x: float
    y: float
    r: float
    terms: tuple[LdaTerm, ...]
    caption: str


@dataclasses.dataclass(frozen=True)
class LdaVis:
    """pyLDAvis's view, ready to draw: the map and the term lists.

    Attributes:
        topics: The topics, in the topic table's order.
        salient: The most salient terms of the collection, shown while no
            topic is chosen.
        origin: Where the axes cross, from 0 to 100.
        height: The drawing's height: the 100 by 100 map, and the size
            legend in a band below it.
        legend: The size legend, concentric bubbles on one base: each
            share's label, its bubble's radius and centre down, and its
            label's place down.
        legend_x: The legend bubbles' centre across.
        legend_text_x: Where the legend's labels start.
        terms_shown: How many terms a list shows.
        lambda_: The relevance weight the lists start with.
        radius: The bubbles' largest radius, for ``app.js``.
    """

    topics: tuple[LdaTopic, ...]
    salient: tuple[LdaTerm, ...]
    origin: tuple[float, float]
    height: float
    legend: tuple[tuple[str, float, float, float], ...]
    legend_x: float
    legend_text_x: float
    terms_shown: int
    lambda_: float
    radius: float

    @property
    def pc1(self) -> tuple[float, float]:
        """Where the first axis's label sits: at its right end."""
        return 98.0, round(self.origin[1] - 1.6, 2)

    @property
    def pc2(self) -> tuple[float, float]:
        """Where the second axis's label sits: at its top."""
        return round(self.origin[0] + 1.6, 2), 4.0

    @property
    def legend_title(self) -> tuple[float, float]:
        """Where the size legend's title sits: right of its labels."""
        return round(self.legend_text_x + 10, 2), round(self.height - 8, 2)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


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
        """The topics and what their library draws, if the run drew them."""
        return self.saved.overview

    @property
    def kind(self) -> str:
        """The algorithm, which picks the views: bertopic, top2vec or lda."""
        return str(self.saved.algorithm)

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
        numbers = [item.topic for item in overview.topics]
        shares = [
            item.prevalence or item.documents / total
            for item in overview.topics
        ]
        margin = map_margin(shares)
        views = []
        for index, item in enumerate(overview.topics):
            share = item.documents / total
            prevalence = shares[index]
            place: dict[str, float] = {}
            if index < len(positions):
                x, y = positions[index]
                place = {
                    "x": round(margin + (100 - 2 * margin) * x, 2),
                    "y": round(margin + (100 - 2 * margin) * (1 - y), 2),
                    "r": bubble(prevalence),
                }
            weights = item.weights or [1.0] * len(item.keywords)
            views.append(
                TopicView(
                    topic=item.topic,
                    index=index,
                    region=region(job_id, index),
                    title=" · ".join(item.keywords[:3]),
                    documents=item.documents,
                    share=share,
                    prevalence=prevalence,
                    bar=round(100 * item.documents / max(largest, 1), 2),
                    keywords=tuple(
                        (word, round(100 * weight, 2))
                        for word, weight in zip(
                            item.keywords, weights, strict=False
                        )
                    ),
                    examples=tuple(item.examples),
                    mixtures=tuple(
                        mixture_bar(parts, index, numbers)
                        for parts in item.mixtures
                    ),
                    cloud=cloud_sizes(item.cloud),
                    **place,
                )
            )
        return views

    def ranked(self, job_id: str) -> list[TopicView]:
        """Return the topics by size, the largest first (Top2Vec's sizes).

        Args:
            job_id: The job's id.

        Returns:
            The topics.
        """
        return sorted(
            self.topics(job_id),
            key=lambda topic: (-topic.documents, topic.index),
        )

    def heatmap(self, job_id: str) -> Heatmap | None:
        """Return the similarity heatmap, in the hierarchy's order if known.

        Args:
            job_id: The job's id.

        Returns:
            The heatmap, or ``None`` without similarities.
        """
        overview = self.overview
        matrix = overview.similarity if overview else None
        if not matrix:
            return None
        names = [topic.topic for topic in self.topics(job_id)]
        order = (
            overview.hierarchy.order
            if overview.hierarchy
            and len(overview.hierarchy.order) == len(names)
            else list(range(len(names)))
        )
        cell, margin = 10, 14
        return Heatmap(
            size=margin + cell * len(names),
            labels=tuple(
                (margin + cell * place + cell / 2, names[index])
                for place, index in enumerate(order)
            ),
            cells=tuple(
                (
                    margin + cell * column,
                    margin + cell * row,
                    matrix[first][second],
                    f"Topics {names[first]} and {names[second]}: "
                    f"{matrix[first][second]:.2f}",
                )
                for row, first in enumerate(order)
                for column, second in enumerate(order)
            ),
        )

    def dendrogram(self, job_id: str) -> Dendrogram | None:
        """Return how the topics merge, ready to draw.

        Leaves are on the left, one row each, labelled with the topics;
        merges reach right as far as the distance at which they join.

        Args:
            job_id: The job's id.

        Returns:
            The dendrogram, or ``None`` without a hierarchy.
        """
        tree = self.overview.hierarchy if self.overview else None
        topics = self.topics(job_id)
        if tree is None or len(tree.order) != len(topics):
            return None
        row, top, left, width = 22.0, 8.0, 300.0, 285.0
        height = tree.height or 1.0

        def x_of(distance: float) -> float:
            return round(left + width * distance / height, 2)

        def y_of(place: float) -> float:
            return round(top + row * place + row / 2, 2)

        axis_y = top + row * len(topics) + 6
        return Dendrogram(
            height=axis_y + 24,
            leaves=tuple(
                (topics[index], y_of(place))
                for place, index in enumerate(tree.order)
            ),
            links=tuple(
                " ".join(
                    f"{x_of(distance)},{y_of(place)}"
                    for distance, place in zip(distances, places, strict=True)
                )
                for distances, places in tree.links
            ),
            ticks=tuple(
                (x_of(height * step), f"{height * step:.2f}")
                for step in (0.0, 0.5, 1.0)
            ),
            axis_y=axis_y,
            label_x=left - 8,
            left=left,
            right=left + width,
        )

    def document_plot(self, job_id: str) -> DocumentPlot | None:
        """Return the documents placed by meaning, ready to draw.

        Args:
            job_id: The job's id.

        Returns:
            The plot, or ``None`` without a document map.
        """
        found = self.overview.document_map if self.overview else None
        if found is None or not found.points:
            return None
        topics = self.topics(job_id)
        points = []
        places: dict[int, list[tuple[float, float]]] = {}
        for (x, y, topic), text in zip(
            found.points, found.excerpts, strict=False
        ):
            px, py = round(3 + 94 * x, 2), round(3 + 94 * (1 - y), 2)
            index = int(topic)
            known = 0 <= index < len(topics)
            points.append(
                DocumentPoint(
                    px, py, topics[index].region if known else "", text
                )
            )
            if known:
                places.setdefault(index, []).append((px, py))
        labels = tuple(
            (topics[index], *_middle(spots))
            for index, spots in sorted(places.items())
        )
        return DocumentPlot(
            points=tuple(points),
            labels=labels,
            drawn=len(points),
            documents=found.documents,
        )

    def trends(self, job_id: str) -> Trends | None:
        """Return the topics over time on one chart, ready to draw.

        Args:
            job_id: The job's id.

        Returns:
            The chart, or ``None`` without topics over time.
        """
        overview = self.overview
        if overview is None or not overview.over_time:
            return None
        labels = overview.time_labels
        topics = self.topics(job_id)
        left, right, top, bottom = 44.0, 588.0, 12.0, 226.0
        peak = max(
            (max(counts, default=0) for counts in overview.over_time),
            default=0,
        )
        ceiling = nice_ceiling(peak)
        columns = len(labels)

        def x_of(column: int) -> float:
            if columns < 2:
                return round((left + right) / 2, 2)
            return round(left + (right - left) * column / (columns - 1), 2)

        def y_of(count: float) -> float:
            return round(bottom - (bottom - top) * count / ceiling, 2)

        trends = []
        for topic, counts, item in zip(
            topics, overview.over_time, overview.topics, strict=False
        ):
            words = item.time_words or [""] * columns
            points = tuple(
                TrendPoint(
                    x_of(column),
                    y_of(count),
                    f"Topic {topic.topic}, {labels[column]}: {count} "
                    f"document{'s' if count != 1 else ''}"
                    + (f" ({words[column]})" if words[column] else ""),
                )
                for column, count in enumerate(counts)
            )
            top_count = max(counts, default=0)
            trends.append(
                Trend(
                    topic=topic,
                    line=" ".join(f"{p.x},{p.y}" for p in points),
                    points=points,
                    peak=top_count,
                    peak_at=labels[counts.index(top_count)]
                    if top_count
                    else "",
                    counts=tuple(counts),
                )
            )
        marks = sorted({0, columns // 2, columns - 1}) if columns else []
        return Trends(
            trends=tuple(trends),
            labels=tuple(labels),
            ticks_x=tuple(
                (
                    x_of(column),
                    labels[column],
                    "start"
                    if column == 0 and columns > 1
                    else "end"
                    if column == columns - 1 and columns > 1
                    else "middle",
                )
                for column in marks
            ),
            ticks_y=tuple(
                (y_of(value), f"{value:,}")
                for value in sorted({0, ceiling // 2, ceiling})
            ),
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )

    def ldavis(self, job_id: str) -> LdaVis | None:
        """Return pyLDAvis's view, ready to draw.

        Args:
            job_id: The job's id.

        Returns:
            The view, or ``None`` if the run has no pyLDAvis data.
        """
        overview = self.overview
        if overview is None or not overview.salient:
            return None
        topics = self.topics(job_id)
        shares = {
            term: " ".join(f"{index}:{share}" for index, share in spread)
            for term, spread in overview.term_topics.items()
        }
        lda_topics = []
        for topic, item in zip(topics, overview.topics, strict=False):
            terms = rank_terms(item.terms, shares)
            lda_topics.append(
                LdaTopic(
                    topic=topic,
                    x=topic.x,
                    y=topic.y,
                    r=topic.r,
                    terms=terms,
                    caption=f"Top {LDAVIS_TERMS} most relevant terms for "
                    f"topic {topic.topic} ({topic.prevalence:.1%} of the "
                    "words)",
                )
            )
        salient_max = max((total for _, total in overview.salient), default=1)
        salient = tuple(
            LdaTerm(
                term=term,
                freq=total,
                total=total,
                logprob=0.0,
                loglift=0.0,
                shares=shares.get(term, ""),
                total_bar=round(100 * total / (salient_max or 1), 2),
                freq_bar=0.0,
                shown=True,
            )
            for term, total in overview.salient[:LDAVIS_TERMS]
        )
        origin = overview.origin or [0.5, 0.5]
        widest = bubble(max(SIZE_LEGEND))
        base = round(100 + 3 + 2 * widest, 2)
        margin = map_margin([topic.prevalence for topic in topics])
        return LdaVis(
            topics=tuple(lda_topics),
            salient=salient,
            origin=(
                round(margin + (100 - 2 * margin) * origin[0], 2),
                round(margin + (100 - 2 * margin) * (1 - origin[1]), 2),
            ),
            height=round(base + 2, 2),
            legend=tuple(
                (
                    f"{share:.0%}",
                    bubble(share),
                    round(base - bubble(share), 2),
                    round(base - 2 * bubble(share), 2),
                )
                for share in SIZE_LEGEND
            ),
            legend_x=round(2 + widest, 2),
            legend_text_x=round(4 + 2 * widest, 2),
            terms_shown=LDAVIS_TERMS,
            lambda_=LDAVIS_LAMBDA,
            radius=BUBBLE_RADIUS,
        )

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def region(job_id: str, index: int) -> str:
    """Return the id that links a topic's marks on the page."""
    return f"{job_id}-topic-{index}"


def bubble(share: float) -> float:
    """Return a bubble's radius, its area in proportion to a share."""
    return round(max(1.2, BUBBLE_RADIUS * math.sqrt(max(share, 0.0))), 2)


def map_margin(shares: list[float]) -> float:
    """Return how far a 100 by 100 map keeps bubbles from its edges.

    Far enough for the largest bubble, within reason.
    """
    largest = max((bubble(share) for share in shares), default=0.0)
    return round(min(25.0, max(8.0, largest + 2)), 2)


def nice_ceiling(value: float) -> int:
    """Return a round number at least as large: 1, 2, 5 times a power of 10.

    Args:
        value: The largest value to show.

    Returns:
        The axis's top; at least 1.
    """
    if value <= 1:
        return 1
    power = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 5, 10):
        if step * power >= value:
            return int(step * power)
    return int(10 * power)


def _middle(spots: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the median point of some points."""
    xs = sorted(x for x, _ in spots)
    ys = sorted(y for _, y in spots)
    return xs[len(xs) // 2], ys[len(ys) // 2]


def cloud_sizes(words: list[list[Any]]) -> tuple[tuple[str, int], ...]:
    """Return a word cloud's words with sizes from 1 to 5, by weight.

    Args:
        words: ``[word, weight]`` pairs, weights between 0 and 1.

    Returns:
        The words in the same order, the heaviest 5 and the lightest 1.
    """
    if not words:
        return ()
    weights = [float(weight) for _, weight in words]
    low, high = min(weights), max(weights)
    span = high - low
    return tuple(
        (str(word), 1 + round(4 * (float(weight) - low) / span) if span else 3)
        for word, weight in words
    )


def mixture_bar(
    parts: list[list[float]], this: int, numbers: list[int | str]
) -> tuple[MixturePart, ...]:
    """Return a document's mixture of topics as the parts of a thin bar.

    Each part is as wide as its share, with a small gap between parts; the
    rest of the document (topics below the floor) is one part, "Other".

    Args:
        parts: ``[topic index, share]`` pairs, the largest first.
        this: The index of the topic the document is listed under.
        numbers: Each topic's number, by index.

    Returns:
        The parts, left to right.
    """
    gap = 0.6
    found = []
    x = 0.0
    shown = [(int(index), float(share)) for index, share in parts]
    rest = 1.0 - sum(share for _, share in shown)
    if rest >= 0.02:
        shown.append((-1, rest))
    for place, (index, share) in enumerate(shown):
        width = max(100 * share - gap, 0.4)
        found.append(
            MixturePart(
                topic=numbers[index] if 0 <= index < len(numbers) else "Other",
                share=share,
                x=round(x, 2),
                width=round(width, 2),
                this=index == this,
                shade=place % 2,
            )
        )
        x += 100 * share
    return tuple(found)


def rank_terms(
    rows: list[list[Any]], shares: dict[str, str]
) -> tuple[LdaTerm, ...]:
    """Return a topic's terms as pyLDAvis first lists them (lambda 1).

    Terms are ranked by relevance at :data:`LDAVIS_LAMBDA` (by probability
    alone); the first :data:`LDAVIS_TERMS` are shown, with bars scaled to
    the largest overall frequency among them. ``app.js`` re-ranks and
    re-scales them when lambda changes.

    Args:
        rows: ``[term, frequency in the topic, frequency, log probability,
            log lift]``.
        shares: Each term's spread over the topics, as "index:share" pairs.

    Returns:
        All the terms, the ones shown first.
    """
    lam = LDAVIS_LAMBDA
    ranked = sorted(
        rows,
        key=lambda row: -(lam * float(row[3]) + (1 - lam) * float(row[4])),
    )
    shown = ranked[:LDAVIS_TERMS]
    scale = max((float(row[2]) for row in shown), default=1.0) or 1.0
    return tuple(
        LdaTerm(
            term=str(term),
            freq=float(freq),
            total=float(total),
            logprob=float(logprob),
            loglift=float(loglift),
            shares=shares.get(str(term), ""),
            total_bar=round(100 * float(total) / scale, 2),
            freq_bar=round(100 * float(freq) / scale, 2),
            shown=place < LDAVIS_TERMS,
        )
        for place, (term, freq, total, logprob, loglift) in enumerate(ranked)
    )
