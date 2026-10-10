"""Knowledge Graph page: graphs of a collection of scientific papers.

The page walks through the steps of
:mod:`textlab.features.knowledge_graph.service`: parse a folder of PDFs
into a corpus with Grobid, extract topics with an LLM, and draw ego and
full corpus graphs. Each step reads the corpus folder, so a step can be
started from a corpus made in an earlier session.
"""

import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from textlab.common import gpu_manager
from textlab.common.progress import Progress
from textlab.features.knowledge_graph import service
from textlab.ui.streamlit.auth import check_token
from textlab.ui.streamlit.components.gpu import free_gpu_for

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(app_dir, "assets", "text_lab_logo.png")
try:
    favicon = Image.open(favicon_path)
    st.set_page_config(
        page_title="Knowledge Graph", page_icon=favicon, layout="wide"
    )
except FileNotFoundError:
    st.set_page_config(page_title="Knowledge Graph", layout="wide")

OLLAMA_BACKEND = "Ollama (Local)"
GPUSTACK_BACKEND = "GPUStack (Remote)"

ABOUT = """
    This tool will:
    1. Extract text and metadata from scientific papers using Grobid
    2. Process the content to identify entities and relationships
    3. Generate a knowledge graph visualization
    4. Allow you to explore and export the results

    **Requirements:**
    - Papers must be in PDF format
    - Place all papers in a single directory
    - Ensure you have read access to the directory
    """


def show_progress(bar: Any, status: Any) -> Callable[[Progress], None]:
    """Return a progress callback that updates a bar and a status line.

    Args:
        bar: A Streamlit progress bar.
        status: A Streamlit placeholder for the message.

    Returns:
        The callback.
    """

    def update(progress: Progress) -> None:
        if progress.fraction is not None:
            bar.progress(progress.fraction)
        status.text(progress.message)

    return update


def show_failure(message: str, exc: Exception) -> None:
    """Show an error with its traceback."""
    st.error(f"{message}: {exc}")
    st.code(traceback.format_exc())


def select_corpus(
    corpora: list[Path], label: str, help_text: str, key: str | None = None
) -> Path:
    """Let the user pick one of several corpus folders.

    Args:
        corpora: The corpus folders.
        label: The select box label.
        help_text: Its help text.
        key: Its widget key.

    Returns:
        The selected folder.
    """
    options = {folder.name: folder for folder in corpora}
    name = st.selectbox(
        label, options=list(options.keys()), help=help_text, key=key
    )
    selected = options[name]
    st.info(f"Selected: `{selected}`")
    return selected


def topics_download(corpus: Path, key: str | None = None) -> None:
    """Offer the corpus table with topics as a JSON download."""
    st.download_button(
        label="Download Corpus with Topics (JSON)",
        data=service.topics_json(corpus),
        file_name="corpus_table.with_topics.json",
        mime="application/json",
        key=key,
    )


# ---------------------------------------------------------------------------
# Step 1: corpus
# ---------------------------------------------------------------------------


def generate_corpus_step(papers_dir: Path, pdf_files: list[Path]) -> None:
    """Show step 1: parse the PDFs into a corpus folder."""
    st.subheader("Output Configuration")
    output_location = st.text_input(
        "Output directory location",
        value=str(papers_dir.parent),
        help=(
            "Directory where the output folder will be created. Default: "
            "same level as input folder."
        ),
    )
    output_path = Path(output_location) / service.corpus_folder_name(
        papers_dir
    )
    st.info(f"Output will be saved to: `{output_path}`")

    st.markdown("### Step 1: Generate Corpus")
    st.markdown(
        "Extract metadata and text from all PDFs. This will skip papers "
        "already processed."
    )
    if not st.button("Generate Corpus", type="primary"):
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        summary = service.build_corpus(
            pdf_files,
            output_path,
            on_progress=show_progress(progress_bar, status_text),
        )
    except Exception as exc:
        show_failure("Processing failed", exc)
        return
    progress_bar.progress(1.0)
    status_text.empty()

    st.success(f"""
    **Corpus Generation Complete!**
    - New papers processed: {summary.processed}
    - Skipped (already processed): {summary.skipped}
    - Errors: {summary.errors}
    - Total papers in corpus: {len(summary.table)}
    - Output saved to: `{output_path}`
    """)
    with st.expander("View Corpus Table Preview"):
        st.dataframe(summary.table.head(10), use_container_width=True)
    csv_path = output_path / service.TABLE_CSV
    if csv_path.exists():
        st.download_button(
            label="Download Corpus Table (CSV)",
            data=csv_path.read_bytes(),
            file_name="corpus_table.csv",
            mime="text/csv",
        )


def rebuild_table_step(papers_dir: Path) -> None:
    """Show step 1b: rebuild the corpus table of an existing corpus."""
    st.markdown("---")
    st.markdown("### Step 1b: Rebuild Corpus Table")
    st.markdown(
        "Rebuild the corpus metadata table from existing processed papers. "
        "Use this to update citation extraction."
    )
    location = st.text_input(
        "Corpus location to rebuild",
        value=str(papers_dir.parent),
        help="Directory containing your *_project_corpus folder(s)",
        key="rebuild_location",
    )
    if not location or not Path(location).is_dir():
        return
    corpora = service.find_corpora(location)
    if not corpora:
        st.warning("No corpus folders found in this location.")
        return

    st.success(f"Found {len(corpora)} corpus/corpora")
    corpus = select_corpus(
        corpora,
        "Select corpus to rebuild",
        "Choose corpus to rebuild metadata table",
        key="rebuild_corpus_select",
    )
    if not st.button(
        "Rebuild Corpus Table", type="secondary", key="rebuild_button"
    ):
        return
    try:
        with st.spinner("Rebuilding corpus metadata table..."):
            table = service.build_corpus_table(corpus)
            st.success(f"""
            **Corpus Table Rebuilt!**
            - Total papers: {len(table)}
            - Updated: corpus_table.csv, corpus_table.jsonl
            - Citations extracted (DOI-verified only)
            """)
            if table.empty:
                return
            with_dois = table["cited_dois"].str.len().gt(0).sum()
            st.info(f"{with_dois} papers have citations with DOIs")
            with st.expander("View Updated Corpus Table"):
                st.dataframe(
                    table[["paper_id", "title", "cited_dois"]].head(10),
                    use_container_width=True,
                )
    except Exception as exc:
        show_failure("Rebuild failed", exc)


# ---------------------------------------------------------------------------
# Step 2: topics
# ---------------------------------------------------------------------------


def choose_model(backend: str) -> str | None:
    """Show the model choice for a backend.

    Args:
        backend: :data:`OLLAMA_BACKEND` or :data:`GPUSTACK_BACKEND`.

    Returns:
        The model, or ``None`` if the backend is not ready.
    """
    if backend == GPUSTACK_BACKEND:
        if not st.session_state.gpustack_api_key:
            st.warning("Please enter your GPUStack API Key in the sidebar.")
            return None
        return service.GPUSTACK_MODEL
    try:
        models = service.ollama_models()
    except Exception as exc:
        st.error(f"Could not connect to Ollama: {exc}")
        return None
    if not models:
        st.error("No models found in Ollama. Please pull a model first.")
        return None
    return st.selectbox("Select Local Model", models, index=0)


def extract_topics_step(default_location: str) -> None:
    """Show step 2: extract topics for a corpus with an LLM."""
    st.markdown("---")
    st.markdown("### Step 2: Extract Topics with LLM")
    st.markdown("Use AI to extract research topics from paper abstracts.")
    location = st.text_input(
        "Corpus directory location",
        value=default_location,
        help="Directory containing your *_project_corpus folder(s)",
    )
    if not location:
        return
    if not Path(location).is_dir():
        st.error("Directory does not exist. Please check the path.")
        return
    corpora = service.find_corpora(location, required=service.TABLE_JSONL)
    if not corpora:
        st.warning(
            "No valid corpus found in this location. Ensure corpus folders "
            "contain `corpus_table.jsonl`."
        )
        return

    st.success(f"Found {len(corpora)} corpus/corpora")
    corpus = select_corpus(
        corpora,
        "Select corpus for topic extraction",
        "Choose which corpus to process",
    )
    if (corpus / service.TOPICS_JSONL).exists():
        st.success("Topics file already exists for this corpus")
        topics_download(corpus, key="download_existing_topics")

    st.markdown("#### Extraction Settings")
    backend = st.radio(
        "Select LLM Backend",
        [OLLAMA_BACKEND, GPUSTACK_BACKEND],
        help=(
            "Use local resources (free, no key) or remote GPU cluster "
            "(requires API key)."
        ),
    )
    model = choose_model(backend)
    if not st.button(
        "Extract Topics", type="secondary", disabled=model is None
    ):
        return

    if backend == OLLAMA_BACKEND:
        free_gpu_for(gpu_manager.LLM, ollama_model=model)
    topic_progress = st.progress(0)
    topic_status = st.empty()
    try:
        if backend == OLLAMA_BACKEND:
            client = service.ollama_client()
        else:
            client = service.gpustack_client(st.session_state.gpustack_api_key)
        service.extract_topics(
            corpus,
            client,
            model,
            on_progress=show_progress(topic_progress, topic_status),
        )
    except Exception as exc:
        show_failure("Topic extraction failed", exc)
        return
    topic_progress.progress(1.0)
    topic_status.empty()
    st.success("""
    **Topic Extraction Complete!**
    - Topics have been extracted and added to the corpus
    - Output file: `corpus_table.with_topics.jsonl`
    """)
    topics_download(corpus)


# ---------------------------------------------------------------------------
# Step 3: graphs
# ---------------------------------------------------------------------------


def graph_step(default_location: str) -> None:
    """Show step 3: ego graphs and the full corpus graph."""
    st.markdown("---")
    st.markdown("### Step 3: Build Knowledge Graph")
    st.markdown(
        "Generate network visualization from corpus with extracted topics."
    )
    location = st.text_input(
        "Knowledge Graph corpus location",
        value=default_location,
        key="kg_location",
        help=(
            "Directory containing corpus with topics (*_project_corpus folder)"
        ),
    )
    if not location:
        return
    if not Path(location).is_dir():
        st.error("Directory does not exist. Please check the path.")
        return
    corpora = service.find_corpora(location, required=service.TOPICS_JSONL)
    if not corpora:
        st.warning(
            "No corpus with topics found. Please complete Step 2 first."
        )
        return

    st.success(f"Found {len(corpora)} corpus/corpora with topics")
    corpus = select_corpus(
        corpora,
        "Select corpus for visualization",
        "Choose corpus with extracted topics",
        key="kg_corpus_select",
    )
    records = service.read_records(corpus / service.TOPICS_JSONL)
    ego_graph_section(corpus, records)
    full_graph_section(corpus, records)


def ego_graph_section(corpus: Path, records: list[dict[str, Any]]) -> None:
    """Show the ego graph of one paper."""
    options = {
        f"{rec.get('paper_id', 'unknown')} - "
        f"{rec.get('pdf_filename', 'unknown.pdf')}": rec
        for rec in records
    }
    label = st.selectbox(
        "Select paper to visualize ego graph",
        options=list(options.keys()),
        help="Choose a paper to see its network neighborhood",
        key="paper_select",
    )
    paper = options[label]
    paper_id = paper.get("paper_id", "unknown")
    st.write(f"**Selected:** {paper.get('title', 'Untitled')}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        include_authors = st.checkbox("Authors", value=True, key="ego_authors")
    with col2:
        include_topics = st.checkbox("Topics", value=True, key="ego_topics")
    with col3:
        include_cited_papers = st.checkbox(
            "Cited Papers", value=True, key="ego_cited_papers"
        )
    with col4:
        include_cited_authors = st.checkbox(
            "Cited Authors", value=True, key="ego_cited_authors"
        )

    if not st.button("Generate Ego Graph", type="primary"):
        return
    try:
        with st.spinner("Building ego graph..."):
            graph, network = service.build_paper_ego_graph(
                paper,
                all_records=records,
                include_topics=include_topics,
                include_authors=include_authors,
                include_cited_papers=include_cited_papers,
                include_cited_authors=include_cited_authors,
            )
            file_name = f"{paper_id}_ego_graph.html"
            html = service.save_graph_html(network, corpus / file_name)
            st.success("Ego graph generated!")
            components.html(html, height=650, scrolling=True)

            st.markdown("**Graph Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes", graph.number_of_nodes())
            with col2:
                st.metric("Edges", graph.number_of_edges())
            with col3:
                st.metric("Authors", service.count_nodes(graph, "author"))
            st.download_button(
                label="Download HTML",
                data=html,
                file_name=file_name,
                mime="text/html",
            )
    except Exception as exc:
        show_failure("Error generating ego graph", exc)


def paper_selection(records: list[dict[str, Any]]) -> None:
    """Show the optional choice of papers for the full corpus graph.

    The choice is kept in ``st.session_state.selected_papers`` (paper id
    to included).
    """
    with st.expander(
        "Advanced: Select Specific Papers (Optional)", expanded=False
    ):
        st.markdown(
            "By default, all papers are included. Check boxes below to "
            "select specific papers only."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            select_all = st.button("Select All", key="select_all_papers")
        with col_b:
            deselect_all = st.button("Deselect All", key="deselect_all_papers")

        if "selected_papers" not in st.session_state or select_all:
            st.session_state.selected_papers = {
                rec.get("paper_id"): True for rec in records
            }
        if deselect_all:
            st.session_state.selected_papers = {
                rec.get("paper_id"): False for rec in records
            }

        st.markdown("**Select papers to include:**")
        num_cols = 3
        cols = st.columns(num_cols)
        for idx, rec in enumerate(records):
            paper_id = rec.get("paper_id", "unknown")
            title = rec.get("title", "Untitled")
            if len(title) > 40:
                display_text = f"{paper_id} - {title[:40]}..."
            else:
                display_text = f"{paper_id} - {title}"
            with cols[idx % num_cols]:
                st.session_state.selected_papers[paper_id] = st.checkbox(
                    display_text,
                    value=st.session_state.selected_papers.get(paper_id, True),
                    key=f"paper_checkbox_{paper_id}",
                )

        selected_count = sum(st.session_state.selected_papers.values())
        st.info(f"Selected: {selected_count} / {len(records)} papers")


def full_graph_section(corpus: Path, records: list[dict[str, Any]]) -> None:
    """Show the graph of all (or the selected) papers."""
    st.markdown("---")
    st.markdown("#### Full Corpus Graph")
    st.markdown(
        "Visualize all papers, authors, and topics together to see "
        "collaboration patterns and topic clusters."
    )
    paper_selection(records)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        include_authors = st.checkbox(
            "Authors", value=True, key="full_authors"
        )
    with col2:
        include_topics = st.checkbox("Topics", value=True, key="full_topics")
    with col3:
        include_cited_papers = st.checkbox(
            "Cited Papers", value=False, key="full_cited_papers"
        )
    with col4:
        include_cited_authors = st.checkbox(
            "Cited Authors", value=False, key="full_cited_authors"
        )
    with col5:
        min_confidence = st.slider(
            "Min Topic Confidence", 0.0, 1.0, 0.0, 0.1, key="min_conf"
        )

    if not st.button("Generate Full Corpus Graph", type="secondary"):
        return
    selected = st.session_state.get("selected_papers", {})
    papers = [
        rec for rec in records if selected.get(rec.get("paper_id"), True)
    ]
    if not papers:
        st.warning("No papers selected. Please select at least one paper.")
        return
    try:
        with st.spinner("Building full corpus graph..."):
            graph, network = service.build_full_corpus_graph(
                papers,
                include_topics=include_topics,
                include_authors=include_authors,
                include_cited_papers=include_cited_papers,
                include_cited_authors=include_cited_authors,
                min_topic_confidence=min_confidence,
            )
            html = service.save_graph_html(
                network, corpus / "full_corpus_graph.html"
            )
            st.success(
                f"Full corpus graph generated with {len(papers)} papers!"
            )
            components.html(html, height=850, scrolling=True)

            st.markdown("**Graph Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Nodes", graph.number_of_nodes())
            with col2:
                st.metric("Total Edges", graph.number_of_edges())
            with col3:
                st.metric("Papers", service.count_nodes(graph, "paper"))
            with col4:
                st.metric("Authors", service.count_nodes(graph, "author"))
            st.download_button(
                label="Download Full Graph HTML",
                data=html,
                file_name="full_corpus_graph.html",
                mime="text/html",
            )
    except Exception as exc:
        show_failure("Error generating full corpus graph", exc)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def papers_section(home: str) -> None:
    """Show the papers folder input and, for a folder with PDFs, the steps."""
    st.subheader("Paper Collection Directory")
    st.markdown(
        "Enter the path to the directory containing your scientific papers "
        "(PDF format)."
    )
    papers_path = st.text_input(
        "Papers Directory Path",
        value=str(Path(home) / "papers"),
        help=(
            "Enter the full path to the directory containing PDF files to "
            "process"
        ),
    )
    if not papers_path:
        return
    papers_dir = Path(papers_path)
    if not papers_dir.exists():
        st.error("Directory does not exist. Please check the path.")
        return
    if not papers_dir.is_dir():
        st.error("The path exists but is not a directory")
        return
    pdf_files = service.list_pdfs(papers_dir)
    if not pdf_files:
        st.warning("No PDF files found in this directory")
        return

    st.success(f"Found {len(pdf_files)} PDF file(s) in the directory")
    with st.expander("View PDF files"):
        for pdf in pdf_files:
            st.write(f"- {pdf.name}")

    generate_corpus_step(papers_dir, pdf_files)
    rebuild_table_step(papers_dir)
    default_location = str(papers_dir.parent)
    extract_topics_step(default_location)
    graph_step(default_location)


def main() -> None:
    """Render the Knowledge Graph page."""
    check_token()
    if "gpustack_api_key" not in st.session_state:
        st.session_state.gpustack_api_key = ""

    st.title("Knowledge Graph Generator")
    st.markdown(
        "Process scientific papers and generate a knowledge graph from them."
    )
    with st.sidebar:
        st.header("API Configuration")
        api_key = st.text_input(
            "GPUStack API Key",
            value=st.session_state.gpustack_api_key,
            type="password",
            help="Required ONLY if you select 'GPUStack' as your backend.",
        )
        if api_key:
            st.session_state.gpustack_api_key = api_key
            st.success("API Key set.")

    try:
        service.ensure_grobid_server()
        grobid_available = True
    except service.GrobidError as exc:
        st.error(f"**Grobid Server Error:** {exc}")
        st.warning(
            "The Knowledge Graph feature requires Grobid to be properly "
            "configured. Please contact support."
        )
        grobid_available = False

    home = os.environ.get("HOME")
    if not home:
        st.error(
            "**Configuration Error:** `HOME` environment variable is not set."
        )
        st.stop()
    if not grobid_available:
        return

    papers_section(home)
    st.markdown("---")
    st.subheader("About Knowledge Graph Generation")
    st.markdown(ABOUT)


if __name__ == "__main__":
    main()
