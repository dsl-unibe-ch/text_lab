"""Recognising the many ways the clustering stack rejects a small corpus.

Top2Vec and BERTopic both reduce document vectors with UMAP and then look for
dense regions with HDBSCAN. Neither can model a handful of documents, and
neither says so in a way a user of the page could act on: the failure surfaces
from deep inside those dependencies, in more than one exception *type*, with a
message that never mentions the corpus at all. The page then shows a traceback.

This module keeps the list of those signatures in one place, so that a new one
is added once rather than per engine, and can be tested without importing
either engine -- both take around half a minute to load.
"""

from __future__ import annotations

#: Fragments of the failures that all mean "there is not enough here to
#: cluster". Matched case-insensitively against the exception message.
_TOO_SMALL_SIGNATURES = (
    # HDBSCAN found no dense region at all, so there is nothing to concatenate.
    "need at least one array to concatenate",
    # UMAP initialises its embedding from the eigenvectors of the neighbour
    # graph and asks for more of them than there are documents. Arrives as a
    # TypeError from scipy rather than a ValueError.
    "k >= n",
    # Every document was labelled an outlier, leaving nothing to describe.
    "found array with 0 sample",
)


def is_corpus_too_small(exc: BaseException) -> bool:
    """Is *exc* one of the ways the clustering stack reports too few documents?"""
    message = str(exc).lower()
    return any(signature in message for signature in _TOO_SMALL_SIGNATURES)


def too_small_error(n_documents: int, algorithm: str) -> ValueError:
    """The message to show instead of a traceback.

    Returned rather than raised so the caller keeps the original exception as
    the ``__cause__``, which is what puts the real reason in the log.
    """
    return ValueError(
        f"{algorithm} could not find any topics in these {n_documents} document(s). "
        f"It discovers topics as clusters of similar documents, so it needs a larger "
        f"and more varied collection than this. Please switch your algorithm to "
        f"**Latent Dirichlet Allocation (LDA)**, which is better suited for small "
        f"datasets."
    )
