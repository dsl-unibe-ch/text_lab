import streamlit as st
import os
import sys
import pathlib
import json
from pathlib import Path
from openai import OpenAI

st.set_page_config(page_title="Knowledge Graph", layout="wide")

# Add parent directory to path to import auth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token
from utils_KG import (
    ensure_grobid_server, 
    GrobidError,
    grobid_process_pdf_to_xml,
    extract_metadata_fields,
    extract_plain_text_from_tei,
    build_corpus_metadata_and_table,
    add_topics_to_corpus_table,
    build_paper_ego_graph,
    build_full_corpus_graph,
)

check_token()

# Initialize OpenAI client for LLM
client = OpenAI(
    base_url="XXX", 
    api_key="XXX"
)

st.title("Knowledge Graph Generator")
st.markdown("Process scientific papers and generate a knowledge graph from them.")

# --- 1. Start Grobid server ---
try:
    ensure_grobid_server()
    grobid_available = True
except GrobidError as e:
    st.error(f"**Grobid Server Error:** {str(e)}")
    st.warning("The Knowledge Graph feature requires Grobid to be properly configured. Please contact support.")
    grobid_available = False

# --- 2. Get user's home directory ---
HOST_HOME = os.environ.get("HOME")

if not HOST_HOME:
    st.error("**Configuration Error:** `HOME` environment variable is not set.")
    st.stop()

# --- 3. Input for papers directory ---
if grobid_available:
    st.subheader("📁 Paper Collection Directory")
    st.markdown("Enter the path to the directory containing your scientific papers (PDF format).")
    # Default path suggestion
    default_path = str(pathlib.Path(HOST_HOME) / "papers")

    papers_path = st.text_input(
        "Papers Directory Path",
        value=default_path,
        help="Enter the full path to the directory containing PDF files to process"
    )

    # --- 4. Validate path ---
    if papers_path:
        papers_dir = pathlib.Path(papers_path)
        
        if papers_dir.exists() and papers_dir.is_dir():
            # Count PDF files
            pdf_files = list(papers_dir.glob("*.pdf"))
            
            if pdf_files:
                st.success(f"Found {len(pdf_files)} PDF file(s) in the directory")
                
                # Show file list in expander
                with st.expander("📄 View PDF files"):
                    for pdf in pdf_files:
                        st.write(f"- {pdf.name}")
                
                # --- 5. Output directory configuration ---
                st.subheader("📂 Output Configuration")
                
                output_name = f"{papers_dir.name}_project_corpus"
                
                output_location = st.text_input(
                    "Output directory location",
                    value=str(papers_dir.parent),
                    help="Directory where the output folder will be created. Default: same level as input folder."
                )
                
                # Build full output path
                output_path = Path(output_location) / output_name
                st.info(f"Output will be saved to: `{output_path}`")
                
                # --- 6. Process button ---
                st.markdown("### Step 1: Generate Corpus")
                st.markdown("Extract metadata and text from all PDFs. This will skip papers already processed.")
                
                if st.button("Generate Corpus", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Create output directory
                        output_path.mkdir(parents=True, exist_ok=True)
                        
                        total_pdfs = len(pdf_files)
                        processed_count = 0
                        skipped_count = 0
                        error_count = 0
                        
                        # Process each PDF
                        for i, pdf_path in enumerate(pdf_files, start=1):
                            paper_id = f"P{i:04d}"
                            paper_out_dir = output_path / paper_id
                            paper_out_dir.mkdir(parents=True, exist_ok=True)
                            
                            metadata_path = paper_out_dir / "metadata.json"
                            
                            # Skip if already processed (metadata.json exists)
                            if metadata_path.exists():
                                status_text.text(f"⏭️ Skipping {pdf_path.name} (already processed)")
                                skipped_count += 1
                                progress_bar.progress(i / total_pdfs)
                                continue
                            
                            # Process with Grobid
                            try:
                                status_text.text(f"🔄 Processing {i}/{total_pdfs}: {pdf_path.name}")
                                
                                # Extract TEI XML
                                tei_xml = grobid_process_pdf_to_xml(pdf_path)
                                (paper_out_dir / "fulltext.tei.xml").write_text(tei_xml, encoding="utf-8")
                                
                                # Extract plain text
                                fulltext = extract_plain_text_from_tei(tei_xml)
                                (paper_out_dir / "fulltext.txt").write_text(fulltext, encoding="utf-8")
                                
                                # Extract metadata
                                metadata = extract_metadata_fields(tei_xml, pdf_path, paper_id)
                                with open(metadata_path, "w", encoding="utf-8") as f:
                                    json.dump(metadata, f, indent=2)
                                
                                processed_count += 1
                                
                            except Exception as e:
                                error_count += 1
                                status_text.text(f"❌ Error processing {pdf_path.name}: {str(e)}")
                                
                                # Save error details
                                import traceback
                                err_text = f"Error processing {pdf_path.name}:\n\n{traceback.format_exc()}"
                                (paper_out_dir / "error.txt").write_text(err_text, encoding="utf-8")
                            
                            # Update progress
                            progress_bar.progress(i / total_pdfs)
                        
                        # Build corpus table
                        status_text.text("Building corpus metadata table...")
                        df = build_corpus_metadata_and_table(output_path)
                        
                        # Success summary
                        progress_bar.progress(1.0)
                        status_text.empty()
                        
                        st.success(f"""
                        **Corpus Generation Complete!**
                        - New papers processed: {processed_count}
                        - Skipped (already processed): {skipped_count}
                        - Errors: {error_count}
                        - Total papers in corpus: {len(df)}
                        - Output saved to: `{output_path}`
                        """)
                        
                        # Show preview of corpus table in expander
                        with st.expander("View Corpus Table Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Download button for CSV
                        csv_path = output_path / "corpus_table.csv"
                        if csv_path.exists():
                            with open(csv_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download Corpus Table (CSV)",
                                    data=f,
                                    file_name="corpus_table.csv",
                                    mime="text/csv"
                                )
                        
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # --- 6b. Rebuild Corpus Table (for existing corpora) ---
                st.markdown("---")
                st.markdown("### Step 1b: Rebuild Corpus Table")
                st.markdown("Rebuild the corpus metadata table from existing processed papers. Use this to update citation extraction.")
                
                rebuild_location = st.text_input(
                    "Corpus location to rebuild",
                    value=str(papers_dir.parent),
                    help="Directory containing your *_project_corpus folder(s)",
                    key="rebuild_location"
                )
                
                if rebuild_location:
                    rebuild_path = Path(rebuild_location)
                    if rebuild_path.exists() and rebuild_path.is_dir():
                        rebuild_corpus_folders = sorted(rebuild_path.glob("*_project_corpus"))
                        
                        if rebuild_corpus_folders:
                            st.success(f"✅ Found {len(rebuild_corpus_folders)} corpus/corpora")
                            
                            rebuild_corpus_options = {f.name: f for f in rebuild_corpus_folders}
                            selected_rebuild_corpus_name = st.selectbox(
                                "Select corpus to rebuild",
                                options=list(rebuild_corpus_options.keys()),
                                help="Choose corpus to rebuild metadata table",
                                key="rebuild_corpus_select"
                            )
                            
                            selected_rebuild_corpus = rebuild_corpus_options[selected_rebuild_corpus_name]
                            st.info(f"📍 Selected: `{selected_rebuild_corpus}`")
                            
                            if st.button("🔄 Rebuild Corpus Table", type="secondary", key="rebuild_button"):
                                try:
                                    with st.spinner("Rebuilding corpus metadata table..."):
                                        df_rebuild = build_corpus_metadata_and_table(selected_rebuild_corpus)
                                        
                                        st.success(f"""
                                        ✅ **Corpus Table Rebuilt!**
                                        - Total papers: {len(df_rebuild)}
                                        - Updated: corpus_table.csv, corpus_table.jsonl
                                        - Citations extracted (DOI-verified only)
                                        """)
                                        
                                        # Show citation stats
                                        papers_with_dois = df_rebuild['cited_dois'].str.len().gt(0).sum()
                                        st.info(f"📊 {papers_with_dois} papers have citations with DOIs")
                                        
                                        with st.expander("View Updated Corpus Table"):
                                            st.dataframe(df_rebuild[['paper_id', 'title', 'cited_dois']].head(10), use_container_width=True)
                                
                                except Exception as e:
                                    st.error(f"❌ Rebuild failed: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                        else:
                            st.warning("⚠️ No corpus folders found in this location.")
                
                # --- 7. Step 2: Extract Topics ---
                st.markdown("---")
                st.markdown("### Step 2: Extract Topics with LLM")
                st.markdown("Use AI to extract research topics from paper abstracts. Requires an existing corpus.")
                
                # Let user point to corpus folder
                default_corpus_location = str(papers_dir.parent)
                
                corpus_location = st.text_input(
                    "Corpus directory location",
                    value=default_corpus_location,
                    help="Directory containing your *_project_corpus folder(s)"
                )
                
                if corpus_location:
                    corpus_path = Path(corpus_location)
                    
                    # Find corpus folders in the specified location
                    if corpus_path.exists() and corpus_path.is_dir():
                        corpus_folders = sorted(corpus_path.glob("*_project_corpus"))
                        valid_corpora = [f for f in corpus_folders if (f / "corpus_table.jsonl").exists()]
                        
                        if valid_corpora:
                            st.success(f"✅ Found {len(valid_corpora)} corpus/corpora")
                            
                            # Let user select which corpus to process
                            corpus_options = {f.name: f for f in valid_corpora}
                            selected_corpus_name = st.selectbox(
                                "Select corpus for topic extraction",
                                options=list(corpus_options.keys()),
                                help="Choose which corpus to process"
                            )
                            
                            selected_corpus = corpus_options[selected_corpus_name]
                            st.info(f"📍 Selected: `{selected_corpus}`")
                            
                            if st.button("🤖 Extract Topics", type="secondary"):
                                topic_progress = st.progress(0)
                                topic_status = st.empty()
                                
                                try:
                                    # Define progress callbacks
                                    def update_progress(current, total):
                                        topic_progress.progress(current / total)
                                    
                                    def update_status(message):
                                        topic_status.text(message)
                                    
                                    # Call topic extraction with progress tracking
                                    add_topics_to_corpus_table(
                                        output_root=selected_corpus,
                                        client=client,
                                        inplace=False,
                                        model="gpt-oss-120b",
                                        progress_callback=update_progress,
                                        status_callback=update_status
                                    )
                                    
                                    topic_progress.progress(1.0)
                                    topic_status.empty()
                                    
                                    st.success("""
                                    ✅ **Topic Extraction Complete!**
                                    - Topics have been extracted and added to the corpus
                                    - Output file: `corpus_table.with_topics.jsonl`
                                    """)
                                    
                                    # Download button for topics file
                                    topics_file = selected_corpus / "corpus_table.with_topics.jsonl"
                                    if topics_file.exists():
                                        with open(topics_file, "rb") as f:
                                            st.download_button(
                                                label="⬇️ Download Corpus with Topics (JSONL)",
                                                data=f,
                                                file_name="corpus_table.with_topics.jsonl",
                                                mime="application/jsonlines"
                                            )
                                
                                except Exception as e:
                                    st.error(f"❌ Topic extraction failed: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                        else:
                            st.warning("⚠️ No valid corpus found in this location. Ensure corpus folders contain `corpus_table.jsonl`.")
                    else:
                        st.error("❌ Directory does not exist. Please check the path.")
                
                # --- 8. Step 3: Build Knowledge Graph ---
                st.markdown("---")
                st.markdown("### Step 3: Build Knowledge Graph")
                st.markdown("Generate network visualization from corpus with extracted topics.")
                
                # Let user point to enriched corpus folder
                kg_location = st.text_input(
                    "Knowledge Graph corpus location",
                    value=default_corpus_location,
                    key="kg_location",
                    help="Directory containing corpus with topics (*_project_corpus folder)"
                )
                
                if kg_location:
                    kg_path = Path(kg_location)
                    
                    if kg_path.exists() and kg_path.is_dir():
                        # Find corpus folders with topics
                        kg_corpus_folders = sorted(kg_path.glob("*_project_corpus"))
                        valid_kg_corpora = [f for f in kg_corpus_folders if (f / "corpus_table.with_topics.jsonl").exists()]
                        
                        if valid_kg_corpora:
                            st.success(f"✅ Found {len(valid_kg_corpora)} corpus/corpora with topics")
                            
                            # Let user select which corpus
                            kg_corpus_options = {f.name: f for f in valid_kg_corpora}
                            selected_kg_corpus_name = st.selectbox(
                                "Select corpus for visualization",
                                options=list(kg_corpus_options.keys()),
                                help="Choose corpus with extracted topics",
                                key="kg_corpus_select"
                            )
                            
                            selected_kg_corpus = kg_corpus_options[selected_kg_corpus_name]
                            st.info(f"📍 Selected: `{selected_kg_corpus}`")
                            
                            # Load corpus data
                            topics_file = selected_kg_corpus / "corpus_table.with_topics.jsonl"
                            import json
                            
                            # Load all records
                            all_records = []
                            with open(topics_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    all_records.append(json.loads(line))
                            
                            # Let user select which paper to visualize
                            # Create display labels with paper_id and filename only
                            paper_display_options = {}
                            for rec in all_records:
                                paper_id = rec.get("paper_id", "unknown")
                                filename = rec.get("pdf_filename", "unknown.pdf")
                                # Format: "P0001 - filename.pdf"
                                display_label = f"{paper_id} - {filename}"
                                paper_display_options[display_label] = rec
                            
                            selected_display_label = st.selectbox(
                                "Select paper to visualize ego graph",
                                options=list(paper_display_options.keys()),
                                help="Choose a paper to see its network neighborhood",
                                key="paper_select"
                            )
                            
                            selected_paper = paper_display_options[selected_display_label]
                            selected_paper_id = selected_paper.get("paper_id", "unknown")
                            paper_title = selected_paper.get("title", "Untitled")
                            st.write(f"**Selected:** {paper_title}")
                            
                            # Options for what to include in ego graph
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                include_authors = st.checkbox("Authors", value=True, key="ego_authors")
                            with col2:
                                include_topics = st.checkbox("Topics", value=True, key="ego_topics")
                            with col3:
                                include_cited_papers = st.checkbox("Cited Papers", value=True, key="ego_cited_papers")
                            with col4:
                                include_cited_authors = st.checkbox("Cited Authors", value=True, key="ego_cited_authors")
                            
                            if st.button("Generate Ego Graph", type="primary"):
                                try:
                                    with st.spinner("Building ego graph..."):
                                        # Build ego graph
                                        nx_graph, pyvis_net = build_paper_ego_graph(
                                            selected_paper,
                                            all_records=all_records,
                                            include_topics=include_topics,
                                            include_authors=include_authors,
                                            include_cited_papers=include_cited_papers,
                                            include_cited_authors=include_cited_authors
                                        )
                                        
                                        # Save HTML to temp file
                                        html_file = selected_kg_corpus / f"{selected_paper_id}_ego_graph.html"
                                        pyvis_net.save_graph(str(html_file))
                                        
                                        # Read and display HTML
                                        with open(html_file, 'r', encoding='utf-8') as f:
                                            html_content = f.read()
                                        
                                        st.success("✅ Ego graph generated!")
                                        
                                        # Display the graph
                                        import streamlit.components.v1 as components
                                        components.html(html_content, height=650, scrolling=True)
                                        
                                        # Show graph statistics
                                        st.markdown("**Graph Statistics:**")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Nodes", nx_graph.number_of_nodes())
                                        with col2:
                                            st.metric("Edges", nx_graph.number_of_edges())
                                        with col3:
                                            st.metric("Authors", len([n for n in nx_graph.nodes if str(n).startswith("author:")]))
                                        
                                        # Download button
                                        st.download_button(
                                            label="💾 Download HTML",
                                            data=html_content,
                                            file_name=f"{selected_paper_id}_ego_graph.html",
                                            mime="text/html"
                                        )
                                        
                                except ImportError:
                                    st.error("❌ Please install required packages: `pip install networkx pyvis`")
                                except Exception as e:
                                    st.error(f"❌ Error generating ego graph: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                            
                            # --- Full Corpus Graph ---
                            st.markdown("---")
                            st.markdown("#### Full Corpus Graph")
                            st.markdown("Visualize all papers, authors, and topics together to see collaboration patterns and topic clusters.")
                            
                            # Advanced: Paper selection
                            with st.expander("🔍 Advanced: Select Specific Papers (Optional)", expanded=False):
                                st.markdown("By default, all papers are included. Check boxes below to select specific papers only.")
                                
                                # Select all / deselect all buttons
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    select_all = st.button("✅ Select All", key="select_all_papers")
                                with col_b:
                                    deselect_all = st.button("❌ Deselect All", key="deselect_all_papers")
                                
                                # Initialize session state for paper selection if not exists
                                if "selected_papers" not in st.session_state:
                                    st.session_state.selected_papers = {rec.get("paper_id"): True for rec in all_records}
                                
                                # Handle select/deselect all
                                if select_all:
                                    st.session_state.selected_papers = {rec.get("paper_id"): True for rec in all_records}
                                if deselect_all:
                                    st.session_state.selected_papers = {rec.get("paper_id"): False for rec in all_records}
                                
                                # Show checkboxes in columns for better layout
                                st.markdown("**Select papers to include:**")
                                num_cols = 3
                                cols = st.columns(num_cols)
                                
                                for idx, rec in enumerate(all_records):
                                    paper_id = rec.get("paper_id", "unknown")
                                    title = rec.get("title", "Untitled")
                                    filename = rec.get("pdf_filename", "unknown.pdf")
                                    
                                    # Display in rotating columns
                                    with cols[idx % num_cols]:
                                        # Checkbox with paper_id and truncated title
                                        display_text = f"{paper_id} - {title[:40]}..." if len(title) > 40 else f"{paper_id} - {title}"
                                        
                                        is_selected = st.checkbox(
                                            display_text,
                                            value=st.session_state.selected_papers.get(paper_id, True),
                                            key=f"paper_checkbox_{paper_id}"
                                        )
                                        st.session_state.selected_papers[paper_id] = is_selected
                                
                                # Show count of selected papers
                                selected_count = sum(st.session_state.selected_papers.values())
                                st.info(f"Selected: {selected_count} / {len(all_records)} papers")
                            
                            # Options for full corpus graph
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                include_authors_full = st.checkbox("Authors", value=True, key="full_authors")
                            with col2:
                                include_topics_full = st.checkbox("Topics", value=True, key="full_topics")
                            with col3:
                                include_cited_papers_full = st.checkbox("Cited Papers", value=False, key="full_cited_papers")
                            with col4:
                                include_cited_authors_full = st.checkbox("Cited Authors", value=False, key="full_cited_authors")
                            with col5:
                                min_confidence = st.slider("Min Topic Confidence", 0.0, 1.0, 0.0, 0.1, key="min_conf")
                            
                            if st.button("🌐 Generate Full Corpus Graph", type="secondary"):
                                try:
                                    with st.spinner("Building full corpus graph..."):
                                        # Filter records based on selection
                                        if "selected_papers" in st.session_state:
                                            filtered_records = [
                                                rec for rec in all_records 
                                                if st.session_state.selected_papers.get(rec.get("paper_id"), True)
                                            ]
                                        else:
                                            filtered_records = all_records
                                        
                                        if not filtered_records:
                                            st.warning("⚠️ No papers selected. Please select at least one paper.")
                                        else:
                                            # Build full corpus graph with filtered records
                                            nx_graph_full, pyvis_net_full = build_full_corpus_graph(
                                                filtered_records,
                                                include_topics=include_topics_full,
                                                include_authors=include_authors_full,
                                                include_cited_papers=include_cited_papers_full,
                                                include_cited_authors=include_cited_authors_full,
                                                min_topic_confidence=min_confidence
                                            )
                                        
                                        # Save HTML to temp file
                                        html_file_full = selected_kg_corpus / "full_corpus_graph.html"
                                        pyvis_net_full.save_graph(str(html_file_full))
                                        
                                        # Read and display HTML
                                        with open(html_file_full, 'r', encoding='utf-8') as f:
                                            html_content_full = f.read()
                                        
                                        st.success(f"✅ Full corpus graph generated with {len(filtered_records)} papers!")
                                        
                                        # Display the graph
                                        import streamlit.components.v1 as components
                                        components.html(html_content_full, height=850, scrolling=True)
                                        
                                        # Show graph statistics
                                        st.markdown("**Graph Statistics:**")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Total Nodes", nx_graph_full.number_of_nodes())
                                        with col2:
                                            st.metric("Total Edges", nx_graph_full.number_of_edges())
                                        with col3:
                                            st.metric("Papers", len([n for n in nx_graph_full.nodes if not str(n).startswith(("author:", "topic:"))]))
                                        with col4:
                                            st.metric("Authors", len([n for n in nx_graph_full.nodes if str(n).startswith("author:")]))
                                        
                                        # Download button
                                        st.download_button(
                                            label="💾 Download Full Graph HTML",
                                            data=html_content_full,
                                            file_name="full_corpus_graph.html",
                                            mime="text/html"
                                        )
                                        
                                except ImportError:
                                    st.error("❌ Please install required packages: `pip install networkx pyvis`")
                                except Exception as e:
                                    st.error(f"❌ Error generating full corpus graph: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())

                        else:
                            st.warning("⚠️ No corpus with topics found. Please complete Step 2 first.")
                    else:
                        st.error("❌ Directory does not exist. Please check the path.")
                    
            else:
                st.warning("⚠️ No PDF files found in this directory")
        
        elif papers_dir.exists() and not papers_dir.is_dir():
            st.error("❌ The path exists but is not a directory")
        else:
            st.error("❌ Directory does not exist. Please check the path.")

    # --- 7. Information section ---
    st.markdown("---")
    st.subheader("ℹ️ About Knowledge Graph Generation")
    st.markdown("""
    This tool will:
    1. Extract text and metadata from scientific papers using Grobid
    2. Process the content to identify entities and relationships
    3. Generate a knowledge graph visualization
    4. Allow you to explore and export the results

    **Requirements:**
    - Papers must be in PDF format
    - Place all papers in a single directory
    - Ensure you have read access to the directory
    """)