"""Build and explore saved paper collections through one project interface."""

import os
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from textlab.common import gpu_manager
from textlab.features.knowledge_graph import service
from textlab.ui.streamlit.auth import check_token
from textlab.ui.streamlit.components.gpu import free_gpu_for
from textlab.ui.streamlit.components.progress import StatusBox

OLLAMA_BACKEND = "In this Text Lab session (Ollama)"
GPUSTACK_BACKEND = "University GPUStack service (API key required)"


def show_failure(message: str, exc: Exception) -> None:
    """Show an actionable error with optional technical details."""
    st.error(f"{message}: {exc}")
    with st.expander("Error details"):
        st.code(traceback.format_exc())


def activate_project(corpus: Path) -> None:
    """Select a project and isolate its graph and widget state."""
    project = str(corpus.resolve())
    st.session_state.kg_graph = None
    st.session_state.kg_project = project


def model_settings(key: str) -> tuple[str, str | None, str]:
    """Collect the topic model and show credentials only when needed.

    Args:
        key: Prefix for this settings panel's widget keys.

    Returns:
        Backend, model name if available, and optional API key.
    """
    st.caption(
        "A language model reads each paper's title and abstract to identify "
        "research topics. Choose where to run this extraction."
    )
    backend = st.radio(
        "Topic extraction mode",
        [OLLAMA_BACKEND, GPUSTACK_BACKEND],
        horizontal=True,
        key=f"{key}_backend",
    )
    if backend == GPUSTACK_BACKEND:
        api_key = st.text_input(
            "GPUStack API key",
            type="password",
            key=f"{key}_api_key",
            help="Required to use the university's GPUStack service.",
        )
        st.caption(
            "Your papers' titles and abstracts are sent to the university's "
            "GPUStack service for topic extraction. They stay on university "
            "infrastructure."
        )
        st.caption(f"Model for topic extraction: {service.GPUSTACK_MODEL}")
        return backend, service.GPUSTACK_MODEL if api_key else None, api_key
    st.caption(
        "Uses the computing resources allocated to your Text Lab session. "
        "No API key is needed; titles and abstracts stay in your session."
    )
    try:
        models = service.ollama_models()
    except Exception as exc:
        st.warning(f"The session model configuration could not be read: {exc}")
        return backend, None, ""
    if not models:
        st.info(
            "No models are configured for this session. Contact Text Lab "
            "support or choose the university GPUStack service."
        )
        return backend, None, ""
    model_key = f"{key}_model"
    if st.session_state.get(model_key) not in models:
        st.session_state[model_key] = models[0]
    model = st.selectbox(
        "Model for topic extraction",
        models,
        key=model_key,
        help="Model choices follow the configuration for this session's GPU.",
    )
    return backend, model, ""


def extract_topics(
    corpus: Path, backend: str, model: str, api_key: str, status: Any
) -> service.TopicSummary:
    """Run topic extraction with the chosen model through the service."""
    if backend == OLLAMA_BACKEND:
        free_gpu_for(gpu_manager.LLM, ollama_model=model)
        client = service.ollama_client()
    else:
        client = service.gpustack_client(api_key)
    return service.extract_topics(corpus, client, model, on_progress=status)


def build_project(
    papers: Path, corpus: Path, backend: str, model: str, api_key: str
) -> None:
    """Prepare papers and topics, preserving reusable PDF extractions."""
    try:
        pdfs = service.list_pdfs(papers)
        if not papers.is_dir() or not pdfs:
            st.error("Choose a cluster folder containing PDF papers.")
            return
        with StatusBox(
            "Building your knowledge graph...",
            complete_label="Your project is ready to explore.",
            caption="Extracted papers are saved as processing progresses.",
        ) as status:
            service.ensure_grobid_server()
            summary = service.build_corpus(pdfs, corpus, on_progress=status)
            activate_project(corpus)
            if summary.table.empty:
                raise ValueError(
                    "No papers could be extracted. Check the PDFs and try "
                    "again; the saved project remains available."
                )
            topics = extract_topics(corpus, backend, model, api_key, status)
            st.session_state.kg_graph = None
        st.success(
            f"{summary.processed} papers extracted · "
            f"{summary.skipped} reused · {summary.errors} failed"
        )
        if summary.errors:
            st.warning(
                "Some PDFs could not be processed. You can explore the "
                "successful papers and retry using Update graph."
            )
        if topics.skipped:
            st.info(
                f"{topics.skipped} papers have no abstract; they are included "
                "without research topics."
            )
    except Exception as exc:
        show_failure(
            "The project could not be completed. Saved extractions can "
            "be reused when you retry",
            exc,
        )


def new_project(home: Path) -> None:
    """Choose uploaded papers or a cluster folder and save one project."""
    source = st.radio(
        "Add papers",
        ["Upload PDF or ZIP", "Use a folder on this cluster"],
        horizontal=True,
        key="kg_paper_source",
    )
    upload = None
    papers = None
    folder_name = None
    ready = False
    if source == "Upload PDF or ZIP":
        upload = st.file_uploader(
            "Upload a PDF or a ZIP of PDF papers",
            type=["pdf", "zip"],
            key="kg_paper_upload",
            help=(
                "ZIPs may contain subfolders. "
                "Each PDF needs a unique filename. ZIP contents have "
                "separate extraction limits. For larger collections, "
                "use a folder on this cluster."
            ),
        )
        project_name = st.text_input(
            "Project name",
            placeholder="Literature review",
            key="kg_upload_name",
        )
        if project_name.strip():
            try:
                folder_name = service.project_folder_name(project_name)
            except ValueError as exc:
                st.error(str(exc))
        ready = upload is not None and folder_name is not None
    else:
        papers_text = st.text_input(
            "PDF folder on this cluster",
            value=str(home / "papers"),
            help=(
                "Enter the full path to a folder containing your PDF "
                "papers on this cluster."
            ),
        )
        papers = Path(papers_text.strip()).expanduser()
        folder_name = service.corpus_folder_name(papers)
        ready = bool(papers_text.strip())
        if ready and papers.is_dir():
            st.caption(f"{len(service.list_pdfs(papers))} PDF papers found")
    save_text = st.text_input(
        "Save project in",
        value=str(home),
        key="kg_save_location",
        help="Choose persistent cluster storage to reuse your work later.",
    )
    corpus = (
        Path(save_text.strip()).expanduser() / folder_name
        if folder_name
        else None
    )
    if corpus is not None and save_text.strip():
        st.caption(f"Project folder: {corpus}")
    if source == "Upload PDF or ZIP":
        st.caption(
            "Uploaded PDFs are temporary. Extracted text, metadata, topics, "
            "and graphs are saved in your project folder for future sessions."
        )
    with st.expander("Advanced settings", expanded=False):
        backend, model, api_key = model_settings("new")
    st.caption(
        "Build extracts your PDFs, identifies research topics, and opens "
        "the graph. Existing PDF extractions are reused automatically; "
        "topics are generated with the selected model."
    )
    if st.button(
        "Build knowledge graph",
        type="primary",
        disabled=not (ready and save_text.strip() and model),
    ):
        if upload is not None:
            try:
                with service.staged_papers(
                    upload.name, upload.getvalue()
                ) as staged:
                    build_project(staged, corpus, backend, model, api_key)
            except (ValueError, OSError) as exc:
                show_failure("The uploaded papers could not be prepared", exc)
        elif papers is not None:
            build_project(papers, corpus, backend, model, api_key)


def open_project(home: Path) -> None:
    """Open saved results without starting Grobid or requiring source PDFs."""
    location = st.text_input(
        "Project folder or folder containing projects", value=str(home)
    )
    if not location.strip():
        return
    folder = Path(location.strip()).expanduser()
    if not folder.is_dir():
        st.info("Enter an existing folder on the cluster.")
        return
    if (folder / service.TABLE_JSONL).is_file():
        corpora = [folder]
    else:
        corpora = service.find_corpora(folder)
    if not corpora:
        st.info("No saved projects found here. Choose another folder.")
        return
    corpus = st.selectbox(
        "Saved project", corpora, format_func=lambda path: path.name
    )
    if st.button("Open project", type="primary"):
        activate_project(corpus)


def project_tools(corpus: Path) -> None:
    """Offer optional updates and maintenance outside the main workflow."""
    with st.expander("Project tools"):
        if not st.toggle("Show processing controls", key=f"{corpus}_tools"):
            st.caption(
                "Update papers, change the topic model, or repair tables."
            )
            return
        st.caption(
            "Update the collection or choose a different model. Saved PDF "
            "extractions are reused; topic extraction processes all papers "
            "and replaces saved topics only after it completes."
        )
        papers_text = st.text_input(
            "PDF folder for additions or retries",
            key=f"{corpus}_update_papers",
            help="Only needed for Update graph, not for opening saved work.",
        )
        backend, model, api_key = model_settings(f"{corpus}_update")
        update, topics = st.columns(2)
        if update.button(
            "Update graph",
            disabled=not (papers_text.strip() and model),
        ):
            build_project(
                Path(papers_text.strip()).expanduser(),
                corpus,
                backend,
                model,
                api_key,
            )
        if topics.button(
            "Re-extract topics",
            disabled=model is None
            or not (corpus / service.TABLE_JSONL).is_file(),
        ):
            try:
                with StatusBox(
                    "Extracting topics...",
                    complete_label="Topics updated.",
                    caption="Saved PDF extractions remain available.",
                ) as status:
                    extract_topics(corpus, backend, model, api_key, status)
                st.session_state.kg_graph = None
            except Exception as exc:
                show_failure("Topic extraction could not be completed", exc)
        if st.button("Rebuild paper table"):
            try:
                service.build_corpus_table(corpus)
                st.success("Paper table rebuilt. Saved topics are unchanged.")
            except Exception as exc:
                show_failure("The paper table could not be rebuilt", exc)


def build_graph(
    corpus: Path,
    records: list[dict[str, Any]],
    *,
    paper_id: str | None = None,
    include_authors: bool = True,
    include_topics: bool = True,
    include_cited_papers: bool = False,
    include_cited_authors: bool = False,
    min_topic_confidence: float = 0.0,
) -> None:
    """Save a graph and retain its display independently of button reruns."""
    try:
        with st.spinner("Building graph..."):
            options = {
                "include_authors": include_authors,
                "include_topics": include_topics,
                "include_cited_papers": include_cited_papers,
                "include_cited_authors": include_cited_authors,
            }
            if paper_id is not None:
                paper = next(
                    record
                    for record in records
                    if record["paper_id"] == paper_id
                )
                graph, network = service.build_paper_ego_graph(
                    paper, all_records=records, **options
                )
                filename = f"{paper_id}_ego_graph.html"
            else:
                graph, network = service.build_full_corpus_graph(
                    records,
                    min_topic_confidence=min_topic_confidence,
                    **options,
                )
                filename = "full_corpus_graph.html"
            html = service.save_graph_html(network, corpus / filename)
            st.session_state.kg_graph = {
                "html": html,
                "filename": filename,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "papers": service.count_nodes(graph, "paper"),
                "label": "Single paper" if paper_id else "Whole collection",
            }
    except Exception as exc:
        show_failure("The graph could not be built", exc)


def explore(corpus: Path, records: list[dict[str, Any]]) -> None:
    """Show one graph with compact controls and optional detailed filters."""
    view = st.radio(
        "Graph view",
        ["Whole collection", "Single paper"],
        horizontal=True,
        key=f"{corpus}_graph_view",
    )
    papers = {record["paper_id"]: record for record in records}
    paper_id = None
    if view == "Single paper":
        paper_id = st.selectbox(
            "Paper",
            list(papers),
            format_func=lambda value: (
                f"{value} - "
                f"{papers[value].get('pdf_filename') or 'unknown.pdf'}"
            ),
            key=f"{corpus}_paper",
        )
    with st.expander("Advanced graph options"):
        authors = st.checkbox("Authors", value=True, key=f"{corpus}_authors")
        topics = st.checkbox("Topics", value=True, key=f"{corpus}_topics")
        cited = st.checkbox("Cited papers", value=False, key=f"{corpus}_cited")
        cited_authors = st.checkbox(
            "Cited authors", value=False, key=f"{corpus}_cited_authors"
        )
        confidence = st.slider(
            "Minimum topic confidence (whole collection)",
            0.0,
            1.0,
            0.0,
            0.1,
            key=f"{corpus}_confidence",
        )
        included = st.multiselect(
            "Papers in the whole collection view",
            list(papers),
            default=list(papers),
            format_func=lambda value: (
                f"{value} - "
                f"{papers[value].get('pdf_filename') or 'unknown.pdf'}"
            ),
            key=f"{corpus}_included",
        )
    selected = (
        records
        if view == "Single paper"
        else [record for record in records if record["paper_id"] in included]
    )
    if not selected:
        st.info("Select at least one paper to show the graph.")
        return
    topic_file = (corpus / service.TOPICS_JSONL).stat()
    selection = (
        str(corpus),
        topic_file.st_mtime_ns,
        topic_file.st_size,
        tuple(record["paper_id"] for record in selected),
        paper_id,
        authors,
        topics,
        cited,
        cited_authors,
        confidence if paper_id is None else 0.0,
    )
    previous = st.session_state.get("kg_graph")
    if previous is None or previous.get("selection") != selection:
        st.session_state.kg_graph = None
        build_graph(
            corpus,
            selected,
            paper_id=paper_id,
            include_authors=authors,
            include_topics=topics,
            include_cited_papers=cited,
            include_cited_authors=cited_authors,
            min_topic_confidence=confidence,
        )
        if st.session_state.get("kg_graph") is not None:
            st.session_state.kg_graph["selection"] = selection
    result = st.session_state.get("kg_graph")
    if result is None:
        return
    st.caption(f"Displayed graph: {result['label']}")
    st.caption(
        "Papers: blue circles · Authors: green triangles · "
        "Categories: purple boxes · Topics: red ellipses · "
        "External cited papers: orange circles · "
        "Cited authors: gold triangles. "
        "In a single-paper graph, cited collection papers are purple circles."
    )
    recenter = """
    <button type="button" aria-label="Recenter graph"
        title="Fit all nodes in view"
        style="position:fixed;top:12px;left:12px;z-index:1000;
               padding:8px 14px;border:1px solid #ccc;border-radius:6px;
               background:white;color:#222;font:14px sans-serif;cursor:pointer"
        onclick="network.fit({animation:{duration:300}})">Recenter</button>
    """
    display_html = result["html"].replace("</body>", recenter + "</body>")
    components.html(display_html, height=750, scrolling=True)
    nodes, edges, count = st.columns(3)
    nodes.metric("Nodes", result["nodes"])
    edges.metric("Connections", result["edges"])
    count.metric("Papers", result["papers"])
    st.download_button(
        "Download graph (HTML)",
        result["html"],
        file_name=result["filename"],
        mime="text/html",
    )


def project_results(corpus: Path) -> None:
    """Read saved project data and expose exploration, papers and topics."""
    st.subheader(corpus.name.removesuffix("_project_corpus"))
    st.caption(f"Saved on the cluster: {corpus}")
    project_tools(corpus)
    source = corpus / service.TABLE_JSONL
    topic_source = corpus / service.TOPICS_JSONL
    try:
        papers = service.read_records(source) if source.is_file() else []
        records = (
            service.read_records(topic_source)
            if topic_source.is_file()
            else []
        )
        if not papers:
            st.info(
                "This project has no paper table yet. Use Project tools to "
                "rebuild the table or retry processing your PDF folder."
            )
            return
        st.caption(
            f"{len(papers)} saved papers · "
            + ("Topics available" if records else "Topic extraction needed")
        )
        if records and {p["paper_id"] for p in papers} != {
            p["paper_id"] for p in records
        }:
            st.warning(
                "The saved topics cover a different set of papers from the "
                "current paper table. Re-extract topics in Project tools "
                "to include the updated collection. The graph below uses "
                "the last completed topic results."
            )
        graph_tab, paper_tab, topic_tab = st.tabs(
            ["Explore", "Papers", "Topics"]
        )
        with graph_tab:
            if records:
                explore(corpus, records)
            else:
                st.info(
                    "Your extracted papers are saved. Open Project tools "
                    "and choose Re-extract topics to prepare the graph."
                )
        with paper_tab:
            columns = [
                "paper_id",
                "title",
                "authors",
                "publication_date",
                "pdf_filename",
            ]
            st.dataframe(
                [
                    {name: paper.get(name, "") for name in columns}
                    for paper in papers
                ],
                use_container_width=True,
                hide_index=True,
            )
            csv = corpus / service.TABLE_CSV
            if csv.is_file():
                st.download_button(
                    "Download papers (CSV)",
                    csv.read_bytes(),
                    file_name=csv.name,
                    mime="text/csv",
                )
        with topic_tab:
            if records:
                rows = [
                    {"Paper": record.get("title", ""), **topic}
                    for record in records
                    for topic in record.get("topics", [])
                ]
                if rows:
                    st.dataframe(
                        rows, use_container_width=True, hide_index=True
                    )
                else:
                    st.info("No topics were found in the saved results.")
                st.download_button(
                    "Download papers and topics (JSON)",
                    service.topics_json(corpus),
                    file_name="papers_with_topics.json",
                    mime="application/json",
                )
            else:
                st.info("Topics have not been extracted yet.")
    except Exception as exc:
        show_failure("Saved project data could not be read", exc)


def main() -> None:
    """Render the project entry point and any selected saved results."""
    st.set_page_config(page_title="Knowledge Graph", layout="wide")
    check_token()
    st.title("Knowledge Graph")
    st.write(
        "Explore connections between scientific papers, authors, and "
        "research topics."
    )
    st.caption(
        "Projects are saved in a cluster folder for reuse across sessions. "
        "Your original PDFs stay where they are."
    )
    home = Path(os.environ.get("HOME", "."))
    if not st.session_state.get("kg_project") or st.toggle(
        "Choose another project"
    ):
        workflow = st.radio(
            "Project",
            ["Create a project", "Open an existing project"],
            horizontal=True,
        )
        if workflow == "Create a project":
            new_project(home)
        else:
            open_project(home)
    if st.session_state.get("kg_project"):
        project_results(Path(st.session_state.kg_project))
    with st.expander("About Knowledge Graph"):
        st.write(
            "Grobid extracts paper metadata, text, and citations. A language "
            "model identifies topics from titles and abstracts. Graphs "
            "connect papers through authors, topics, categories, "
            "and citations."
        )
        st.caption(
            "PDFs already extracted are recognised by filename and reused. "
            "Changing a PDF without changing its filename does not trigger "
            "new extraction. Saved projects remain until you delete them."
        )


if __name__ == "__main__":
    main()
