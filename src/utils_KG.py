import subprocess
import socket
import time
import sys
import os
import requests
import xml.etree.ElementTree as ET
import re
import json
from pathlib import Path
import pandas as pd
from openai import OpenAI

GROBID_HOST = "127.0.0.1"
GROBID_PORT = 8070 
GROBID_URL = f"http://{GROBID_HOST}:{GROBID_PORT}/api/processFulltextDocument"

# --- CHANGE 1: Update path to match the bound storage path ---
# Since /storage is bound, we can access the SIF directly at this path
GROBID_CONTAINER = os.getenv("GROBID_CONTAINER", "/storage/research/dsl_shared/solutions/ondemand/text_lab/container/grobid_0.8.2.sif")

# Ensure tmp dir is in a writable location (usually HOME in OOD)
GROBID_TMP = os.getenv("GROBID_TMP", os.path.join(os.environ.get("HOME", os.getcwd()), "grobid-tmp"))

NS = {"tei": "http://www.tei-c.org/ns/1.0"}




# System prompt for LLM topic extraction
SYSTEM_PROMPT = """
You are an expert research curator.

Your task: read the provided scientific text (title, abstract or excerpt) 
and extract high-quality, hierarchical research topics.

Output requirements:
- Return ONLY valid JSON (no explanation, no intro, no markdown).
- Use this exact schema:

{
  "topics": [
    { 
      "category": "string",
      "label": "string", 
      "confidence": 0.0, 
      "rationale": "string" 
    }
  ]
}

Rules:
- Provide 3–8 topics.
- Each topic has TWO levels:
  * "category": A BROAD research area (e.g., "Machine Learning", "Climate Science", "Medical Imaging", "Neuroscience")
  * "label": A SPECIFIC topic within that category (2–6 words)
- Categories should be general enough that multiple papers could share them.
- Labels must be specific to the scientific content of this paper.
- Avoid generic words ("methods", "results", "datasets", "analysis", "study", "paper", "experiment").
- Avoid meta-topics ("limitations", "future work", "introduction section").
- Do not mention humans or animals unless they are central to the research question.
- Confidence should be between 0 and 1.
- Rationales must be one sentence each.

Example output:
{
  "topics": [
    { "category": "Computer Vision", "label": "Object Detection Networks", "confidence": 0.95, "rationale": "Paper focuses on improving YOLO architecture." },
    { "category": "Deep Learning", "label": "Transformer Architectures", "confidence": 0.85, "rationale": "Uses attention mechanisms extensively." }
  ]
}
"""

class GrobidError(Exception):
    pass

def _port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def ensure_grobid_server():
    """Ensure Grobid server is running, start it if not."""
    
    # 1. Check if already running (Fast path)
    if _port_open(GROBID_HOST, GROBID_PORT):
        print(f"✔ Grobid is already running on port {GROBID_PORT}")
        return

    # 2. Prepare directories
    os.makedirs(GROBID_TMP, exist_ok=True)
    
    # 3. Check for Apptainer inside the current container
    apptainer_cmd = None
    for cmd in ["apptainer", "singularity"]:
        # shutil.which is safer than subprocess run for checking existence
        import shutil
        if shutil.which(cmd):
            apptainer_cmd = cmd
            break
    
    # --- CHANGE 2: Robust Error Message with Manual Workaround ---
    if apptainer_cmd is None:
        raise GrobidError(
            f"""
            Auto-start failed: The 'apptainer' command is not available inside this container.
            
            WORKAROUND:
            1. Keep this app open.
            2. Go to your Open OnDemand Dashboard -> Clusters -> Shell Access.
            3. Run this command in the terminal to start Grobid manually on the same node:
            
            apptainer exec --bind /storage:/storage {GROBID_CONTAINER} bash -c "cd /opt/grobid && ./grobid-service/bin/grobid-service"
            """
        )

    if not os.path.exists(GROBID_CONTAINER):
        raise GrobidError(f"Grobid container SIF not found at: {GROBID_CONTAINER}")

    # 4. Build command (using the bound paths)
    grobid_command = [
        apptainer_cmd, "exec",
        # We need to bind the temp directory from the host (inner container view)
        "-B", f"{GROBID_TMP}:/opt/grobid/grobid-home/tmp",
        # Pass storage bind through if necessary, though usually inherited if configured on host
        "--env", "GROBID_HOME=/opt/grobid/grobid-home",
        GROBID_CONTAINER,
        "bash", "-c", "cd /opt/grobid && ./grobid-service/bin/grobid-service"
    ]

    print(f"🚀 Attempting to start Grobid: {' '.join(grobid_command)}")

    # 5. Spawn the Grobid service
    try:
        subprocess.Popen(
            grobid_command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True # Detach process so it survives page refreshes
        )
    except Exception as e:
        raise GrobidError(f"Failed to start Grobid container: {e}")

    # 6. Wait loop
    print("⏳ Waiting for Grobid to become responsive...")
    for _ in range(60): 
        if _port_open(GROBID_HOST, GROBID_PORT):
            print("✔ Grobid started successfully!")
            return
        time.sleep(1)

    raise GrobidError("Grobid server started but port 8070 did not open within 60 seconds.")


def grobid_process_pdf_to_xml(pdf_path):
    """
    Send a PDF to Grobid and return the TEI XML as a string.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: TEI XML content
        
    Raises:
        GrobidError: If processing fails
    """
    ensure_grobid_server()
    
    with open(pdf_path, "rb") as f:
        response = requests.post(
            GROBID_URL,
            files={"input": f},
            timeout=60
        )
    
    if response.status_code == 200:
        return response.text
    else:
        raise GrobidError(f"Grobid returned status {response.status_code}: {response.text}")


def extract_metadata_fields(tei_xml, pdf_path, paper_id):
    """
    Extract metadata fields from TEI XML.
    
    Args:
        tei_xml: TEI XML string from Grobid
        pdf_path: Path to original PDF
        paper_id: Unique identifier for the paper
        
    Returns:
        dict: Metadata including title, authors, abstract, keywords, DOI, citations, etc.
    """
    root = ET.fromstring(tei_xml.encode('utf-8'))
    
    metadata = {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path)
    }
    
    # Extract title
    title_elem = root.find(".//tei:titleStmt/tei:title[@type='main']", NS)
    metadata["title"] = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
    
    # Extract authors - try both paths
    authors = []
    
    # Path 1: titleStmt (for paper metadata in some cases)
    for author in root.findall(".//tei:titleStmt/tei:author", NS):
        persName = author.find("tei:persName", NS)
        if persName is not None:
            forename = persName.find("tei:forename[@type='first']", NS)
            surname = persName.find("tei:surname", NS)
            if forename is not None and surname is not None:
                authors.append(f"{forename.text} {surname.text}")
    
    # Path 2: sourceDesc/biblStruct (more common for paper authors)
    if not authors:
        for author in root.findall(".//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author", NS):
            persName = author.find("tei:persName", NS)
            if persName is not None:
                forename = persName.find("tei:forename[@type='first']", NS)
                surname = persName.find("tei:surname", NS)
                if forename is not None and surname is not None:
                    authors.append(f"{forename.text} {surname.text}")
    
    metadata["authors"] = authors
    metadata["n_authors"] = len(authors)
    
    # Extract abstract
    abstract_elem = root.find(".//tei:abstract/tei:div/tei:p", NS)
    metadata["abstract"] = abstract_elem.text.strip() if abstract_elem is not None and abstract_elem.text else ""
    
    # Extract keywords
    keywords = []
    for term in root.findall(".//tei:keywords/tei:term", NS):
        if term.text:
            keywords.append(term.text.strip())
    metadata["keywords"] = keywords
    
    # Extract DOI
    doi_elem = root.find(".//tei:idno[@type='DOI']", NS)
    metadata["DOI"] = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else ""
    
    # Extract journal info
    journal_elem = root.find(".//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:title", NS)
    metadata["journal"] = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else ""
    
    # Extract publication date
    date_elem = root.find(".//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:imprint/tei:date[@type='published']", NS)
    metadata["publication_date"] = date_elem.get("when", "") if date_elem is not None else ""
    
    # Extract citations (cited papers)
    citations = []
    cited_authors = []
    cited_dois = []
    
    for bibl in root.findall(".//tei:text/tei:back//tei:listBibl/tei:biblStruct", NS):
        citation = {}
        
        # Citation ID
        citation["citation_id"] = bibl.get("{http://www.w3.org/XML/1998/namespace}id", "")
        
        # Title
        title_elem = bibl.find(".//tei:analytic/tei:title[@type='main']", NS)
        if title_elem is None:
            title_elem = bibl.find(".//tei:monogr/tei:title", NS)
        citation["title"] = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        
        # Authors
        citation_authors = []
        for author in bibl.findall(".//tei:analytic/tei:author", NS):
            persName = author.find("tei:persName", NS)
            if persName is not None:
                forename = persName.find("tei:forename[@type='first']", NS)
                middle = persName.find("tei:forename[@type='middle']", NS)
                surname = persName.find("tei:surname", NS)
                
                if forename is not None and surname is not None:
                    if middle is not None and middle.text:
                        citation_authors.append(f"{forename.text} {middle.text} {surname.text}")
                    else:
                        citation_authors.append(f"{forename.text} {surname.text}")
                elif surname is not None:
                    citation_authors.append(surname.text)
        
        citation["authors"] = citation_authors
        cited_authors.extend(citation_authors)
        
        # DOI
        doi_elem = bibl.find(".//tei:idno[@type='DOI']", NS)
        doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else ""
        citation["DOI"] = doi
        if doi:
            cited_dois.append(doi)
        
        # Year
        date_elem = bibl.find(".//tei:monogr/tei:imprint/tei:date[@type='published']", NS)
        citation["year"] = date_elem.get("when", "")[:4] if date_elem is not None else ""
        
        citations.append(citation)
    
    metadata["citations"] = citations
    metadata["n_citations"] = len(citations)
    metadata["cited_dois"] = cited_dois
    metadata["cited_authors"] = list(set(cited_authors))  # Unique authors
    
    return metadata


def extract_plain_text_from_tei(tei_xml):
    """
    Extract plain text from TEI XML body.
    
    Args:
        tei_xml: TEI XML string from Grobid
        
    Returns:
        str: Plain text content
    """
    root = ET.fromstring(tei_xml.encode('utf-8'))
    
    # Find the body
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return ""
    
    # Remove references and figures
    for ref in body.findall(".//tei:ref", NS):
        if ref.text:
            ref.text = ""
        ref.tail = ref.tail or ""
    
    for parent in body.iter():
    # list(...) to avoid modifying the list while iterating
        for figure in list(parent.findall("tei:figure", NS)):
            parent.remove(figure)
    
    # Extract text from all paragraphs
    paragraphs = []
    for p in body.findall(".//tei:p", NS):
        text = "".join(p.itertext()).strip()
        text = re.sub(r'\s+', ' ', text)
        if text:
            paragraphs.append(text)
    
    return "\n\n".join(paragraphs)


def build_corpus_metadata_and_table(project_root, metadata_filename="corpus_metadata.json"):
    """
    Build corpus metadata and table from processed PDFs.
    
    Args:
        project_root: Root directory containing P#### folders
        metadata_filename: Name for metadata JSON file
        
    Returns:
        pd.DataFrame: Corpus table with all metadata
    """
    project_path = Path(project_root)
    
    # Scan P#### folders and collect metadata
    metadata_list = []
    for paper_dir in sorted(project_path.glob("P*")):
        if not paper_dir.is_dir():
            continue
            
        metadata_file = paper_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                metadata_list.append(meta)
    
    # Save collected metadata to corpus_metadata.json
    metadata_path = project_path / metadata_filename
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    
    print(f"✔ Wrote {metadata_filename} with {len(metadata_list)} papers")
    
    # Create DataFrame
    rows = []
    for meta in metadata_list:
        # Handle authors - could be list of dicts or list of strings
        authors = meta.get("authors", [])
        if authors and isinstance(authors[0], dict):
            authors_str = "; ".join([a.get("full_name", "") for a in authors])
        else:
            authors_str = "; ".join(authors) if isinstance(authors, list) else str(authors)
        
        # Extract filename from pdf_path
        pdf_path = meta.get("pdf_path", "") or meta.get("filename", "")
        pdf_filename = Path(pdf_path).name if pdf_path else ""
        
        row = {
            "paper_id": meta.get("paper_id", ""),
            "title": meta.get("title", ""),
            "authors": authors_str,
            "n_authors": len(authors) if isinstance(authors, list) else 0,
            "abstract": meta.get("abstract", ""),
            "keywords": "; ".join(meta.get("keywords", [])) if isinstance(meta.get("keywords"), list) else "",
            "DOI": meta.get("DOI", "") or meta.get("doi", ""),
            "journal": meta.get("journal", ""),
            "publication_date": meta.get("publication_date", "") or str(meta.get("year", "")),
            "pdf_path": pdf_path,
            "pdf_filename": pdf_filename,
            "n_citations": meta.get("n_citations", 0) or len(meta.get("citations", [])),
            "citations": json.dumps(meta.get("citations", [])),
            # Extract cited titles (all citations have titles)
            "cited_titles": "; ".join([c.get("title", "").strip() for c in meta.get("citations", []) if c.get("title", "").strip()]),
            # Extract cited DOIs (only when available, semicolon-separated)
            "cited_dois": "; ".join([c.get("doi", "") or c.get("DOI", "") for c in meta.get("citations", []) if c.get("doi") or c.get("DOI")]),
            # Extract all cited authors (from all citations, not just those with DOIs)
            "cited_authors": "; ".join([
                author 
                for c in meta.get("citations", [])
                for author in (c.get("authors", []) if isinstance(c.get("authors"), list) else [])
            ])
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV and JSONL
    csv_path = project_path / "corpus_table.csv"
    jsonl_path = project_path / "corpus_table.jsonl"
    
    df.to_csv(csv_path, index=False)
    df.to_json(jsonl_path, orient='records', lines=True)
    
    print(f"✔ Wrote corpus_table.csv with {len(df)} rows")
    print(f"✔ Wrote corpus_table.jsonl")
    
    return df


def run_llm(
    messages,
    client,
    model="gpt-oss-120b",
    temperature=0.2,
    top_p=0.9,
    max_tokens=16384,
    retries=2
):
    """
    Safe LLM call wrapper with retry logic.
    
    Args:
        messages: List of message dicts for OpenAI API
        client: OpenAI client instance
        model: Model name to use
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens in response
        retries: Number of retry attempts on failure
        
    Returns:
        str: LLM response text
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    last_exc = None

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_exc = e
            if attempt < retries:
                print(f"⚠️ LLM error, retrying ({attempt+1}/{retries})…")
            else:
                raise RuntimeError(
                    f"LLM failed after {retries+1} attempts: {e}"
                )

    raise last_exc


def _build_messages(title, abstract):
    """
    Create OpenAI-compatible messages with system prompt for topic extraction.
    
    Args:
        title: Paper title
        abstract: Paper abstract
        
    Returns:
        list: Message dicts for OpenAI API
    """
    user_content = f"TITLE: {title.strip()}\n\nABSTRACT_OR_TEXT:\n{abstract.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _sanitize_and_validate_topics(raw):
    """
    Convert raw LLM output into validated topic structure.
    
    Args:
        raw: Raw LLM response string
        
    Returns:
        dict: Validated topics dict with schema:
              {"topics": [{"category": str, "label": str, "confidence": float, "rationale": str}, ...]}
              
    Raises:
        ValueError: If output doesn't match expected schema
    """
    raw = raw.strip()

    # Remove accidental markdown fences like ```json ... ```
    if raw.startswith("```"):
        raw = raw.strip("` \n")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    # Try strict JSON; fallback to extracting text between first { and last }
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            obj = json.loads(raw[first:last+1])
        else:
            raise

    # Minimal schema check
    if not isinstance(obj, dict) or "topics" not in obj or not isinstance(obj["topics"], list):
        raise ValueError("Model output missing top-level 'topics' list.")

    cleaned = []
    for t in obj["topics"]:
        if not isinstance(t, dict):
            continue
        category = (t.get("category") or "").strip()
        label = (t.get("label") or "").strip()
        rat = (t.get("rationale") or "").strip()
        try:
            conf = float(t.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        if label:
            # Use "General" as fallback if no category provided
            if not category:
                category = "General"
            cleaned.append({
                "category": category,
                "label": label, 
                "confidence": conf, 
                "rationale": rat
            })

    # Enforce 3–8 topics if possible; clip if too many
    if len(cleaned) > 8:
        cleaned = cleaned[:8]

    return {"topics": cleaned}


def add_topics_to_corpus_table(output_root, client, inplace=False, model="gpt-oss-120b", 
                               progress_callback=None, status_callback=None):
    """
    Read corpus_table.jsonl, call LLM for each record's abstract, and add topics.
    
    Args:
        output_root: Path to project directory containing corpus_table.jsonl
        client: OpenAI client instance
        inplace: If True, replace original file (with backup); if False, create new file
        model: Model name to use
        progress_callback: Optional function(current, total) to report progress
        status_callback: Optional function(message) to report status text
        
    Returns:
        None (writes output file)
    """
    from datetime import datetime, timezone
    
    output_root = Path(output_root)
    inp = output_root / "corpus_table.jsonl"
    if not inp.exists():
        raise FileNotFoundError(f"Missing input file: {inp}")

    tmp_out = output_root / "corpus_table.with_topics.tmp.jsonl"
    final_out = output_root / "corpus_table.with_topics.jsonl"

    processed, skipped = 0, 0
    
    # First pass: count total records
    total_records = sum(1 for _ in inp.open("r", encoding="utf-8"))
    current = 0

    with inp.open("r", encoding="utf-8") as fin, tmp_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            current += 1
            rec = json.loads(line)
            title = rec.get("title", "")
            abstract = (rec.get("abstract") or "").strip()
            paper_id = rec.get("paper_id", "unknown")

            if abstract:
                # Update status
                if status_callback:
                    status_callback(f"🤖 Processing {current}/{total_records}: {paper_id} - {title[:50]}...")
                
                # Direct LLM call
                messages = _build_messages(title, abstract)
                raw = run_llm(messages=messages, client=client, model=model, temperature=0.2, top_p=0.9)
                topics_obj = _sanitize_and_validate_topics(raw)

                rec["topics"] = topics_obj["topics"]
                rec["topics_ts"] = datetime.now(timezone.utc).isoformat()
                processed += 1
            else:
                # No abstract: keep record and mark it
                if status_callback:
                    status_callback(f"⏭️ Skipping {current}/{total_records}: {paper_id} (no abstract)")
                rec["topics"] = []
                rec["topics_note"] = "no_abstract"
                skipped += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            # Update progress
            if progress_callback:
                progress_callback(current, total_records)

    # Atomic finalize
    tmp_out.replace(final_out)

    if inplace:
        # Keep a backup of the original and swap
        backup = output_root / "corpus_table.backup.jsonl"
        inp.replace(backup)
        final_out.replace(inp)

    print("---------------------------------------------------")
    print("Topics enrichment completed.")
    print(f"Processed (with abstract): {processed}")
    print(f"Skipped (no abstract):     {skipped}")
    print(f"Output file:               {inp if inplace else final_out}")
    print("---------------------------------------------------")


def build_paper_ego_graph(paper_record, all_records=None, include_topics=True, 
                         include_authors=True, include_cited_papers=True,
                         include_cited_authors=True):
    """
    Build an ego graph centered on a single paper.
    
    The ego graph shows the paper's immediate network neighborhood:
    - The paper itself (center node)
    - Its authors (if include_authors=True)
    - Its topics (if include_topics=True)
    - Cited papers (if include_cited_papers=True)
    - Cited authors (if include_cited_authors=True)
    
    Parameters
    ----------
    paper_record : dict
        Single paper record from corpus_table.with_topics.jsonl
    all_records : list of dict, optional
        All paper records (needed to find cited papers within corpus)
    include_topics : bool
        Add topic nodes connected to the paper
    include_authors : bool
        Add author nodes connected to the paper
    include_citations : bool
        Add citation edges to other papers in corpus
    
    Returns
    -------
    tuple (networkx.DiGraph, pyvis.network.Network)
        - nx_graph: NetworkX directed graph
        - net: Pyvis Network ready for visualization
    """
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError(
            "Please install required packages: pip install networkx pyvis"
        ) from e
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Extract paper info
    paper_id = paper_record.get("paper_id", "unknown")
    title = paper_record.get("title", "Untitled")
    doi = paper_record.get("DOI", "") or paper_record.get("doi", "")
    
    # Use DOI as node ID if available, otherwise use paper_id
    node_id = f"doi:{doi}" if doi else paper_id
    
    # Add center paper node
    G.add_node(
        node_id,
        label=paper_id,  # Show paper_id as label
        title=f"{title}\nDOI: {doi}" if doi else title,  # Full title on hover
        node_type="paper",
        color="#4A90E2",  # Blue for paper
        size=30,
        shape="dot"
    )
    
    # Add authors
    if include_authors:
        authors_raw = paper_record.get("authors", [])
        
        # Handle both string (semicolon-separated) and list formats
        if isinstance(authors_raw, str):
            # Split by semicolon and clean whitespace
            authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
        elif isinstance(authors_raw, list):
            authors = authors_raw
        else:
            authors = []
        
        for author in authors:
            author_name = author if isinstance(author, str) else author.get("name", "Unknown")
            G.add_node(
                f"author:{author_name}",
                label=author_name,
                title=f"Author: {author_name}",
                node_type="author",
                color="#50C878",  # Green for authors
                size=15,
                shape="triangle"
            )
            # Edge from paper to author
            G.add_edge(node_id, f"author:{author_name}", label="authored_by")
    
    # Add topics (both categories and specific topics)
    if include_topics:
        topics = paper_record.get("topics", [])
        
        # Track categories to avoid duplicates
        added_categories = set()
        
        for topic in topics:
            category = topic.get("category", "General")
            topic_label = topic.get("label", "unknown topic")
            confidence = topic.get("confidence", 0.5)
            
            # Add category node (broad topic) - only once per unique category
            category_id = f"category:{category}"
            if category_id not in added_categories:
                added_categories.add(category_id)
                if category_id not in G.nodes:
                    G.add_node(
                        category_id,
                        label=category,
                        title=f"Category: {category}",
                        node_type="category",
                        color="#9B59B6",  # Purple for categories
                        size=25,  # Larger for categories
                        shape="box"
                    )
                # Connect paper to category
                G.add_edge(node_id, category_id, label="in_category")
            
            # Add specific topic node
            topic_id = f"topic:{topic_label}"
            if topic_id not in G.nodes:
                G.add_node(
                    topic_id,
                    label=topic_label,
                    title=f"Topic: {topic_label}\nCategory: {category}\nConfidence: {confidence:.2f}",
                    node_type="topic",
                    color="#FF6B6B",  # Red for specific topics
                    size=10 + confidence * 15,  # Size by confidence
                    shape="ellipse"
                )
            
            # Connect paper to specific topic
            G.add_edge(node_id, topic_id, label="has_topic", weight=confidence)
            
            # Connect specific topic to its category
            G.add_edge(topic_id, category_id, label="belongs_to", style="dashed")
    
    # Add cited papers
    if include_cited_papers:
        # Build lookup of titles to records in our corpus (lowercase for matching)
        title_to_record = {}
        if all_records:
            for rec in all_records:
                rec_title = rec.get("title", "").strip().lower()
                if rec_title:
                    title_to_record[rec_title] = rec
        
        # Handle cited_titles as either string (semicolon-separated) or list
        cited_titles_raw = paper_record.get("cited_titles", "")
        if isinstance(cited_titles_raw, str):
            cited_titles = [title.strip() for title in cited_titles_raw.split(";") if title.strip()]
        else:
            cited_titles = cited_titles_raw if cited_titles_raw else []
        
        # Get original citations array to access DOIs
        citations_raw = paper_record.get("citations", "")
        if isinstance(citations_raw, str):
            import json
            try:
                citations = json.loads(citations_raw)
            except:
                citations = []
        else:
            citations = citations_raw if isinstance(citations_raw, list) else []
        
        # Build title-to-doi lookup from citations
        title_to_doi = {}
        for cit in citations:
            cit_title = cit.get("title", "").strip()
            cit_doi = (cit.get("DOI", "") or cit.get("doi", "")).strip()
            if cit_title:
                title_to_doi[cit_title] = cit_doi
        
        # Add all cited papers (whether in our corpus or not)
        for cited_title in cited_titles:
            cited_title_clean = cited_title.strip()
            cited_title_lower = cited_title_clean.lower()
            cited_node_id = f"paper:{cited_title_clean[:50]}"  # Use title prefix as node ID
            
            # Get DOI if available
            cited_doi = title_to_doi.get(cited_title, "")
            
            # Check if cited paper is in our corpus
            if cited_title_lower in title_to_record:
                cited_rec = title_to_record[cited_title_lower]
                cited_paper_id = cited_rec.get("paper_id", "unknown")
                node_color = "#9B59B6"  # Purple for cited papers in corpus
                node_label = cited_paper_id
                node_title_text = f"{cited_title_clean}\n(in corpus: {cited_paper_id})"
                if cited_doi:
                    node_title_text += f"\nDOI: {cited_doi}"
            else:
                # Cited paper NOT in our corpus
                node_color = "#FFA500"  # Orange for external cited papers
                # Truncate long titles for label
                node_label = cited_title_clean[:30] + "..." if len(cited_title_clean) > 30 else cited_title_clean
                node_title_text = f"{cited_title_clean}\n(not in corpus)"
                if cited_doi:
                    node_title_text += f"\nDOI: {cited_doi}"
            
            # Add cited paper node if not already present
            if cited_node_id not in G.nodes:
                G.add_node(
                    cited_node_id,
                    label=node_label,
                    title=node_title_text,
                    node_type="cited_paper",
                    color=node_color,
                    size=15,
                    shape="dot"
                )
            
            # Add citation edge
            G.add_edge(node_id, cited_node_id, label="cites", color="#999999")
    
    # Add cited authors
    if include_cited_authors:
        # Get cited authors from record
        cited_authors_raw = paper_record.get("cited_authors", "")
        if isinstance(cited_authors_raw, str):
            cited_authors = [author.strip() for author in cited_authors_raw.split(";") if author.strip()]
        else:
            cited_authors = cited_authors_raw if cited_authors_raw else []
        
        for author_name in cited_authors:
            author_id = f"cited_author:{author_name}"
            
            # Add cited author node if not already present
            if author_id not in G.nodes:
                G.add_node(
                    author_id,
                    label=author_name,
                    title=f"Cited Author: {author_name}",
                    node_type="cited_author",
                    color="#FFD700",  # Gold for cited authors
                    size=12,
                    shape="triangle"
                )
            
            # Connect cited author to main paper
            G.add_edge(node_id, author_id, label="cites_work_by", color="#CCCCCC", style="dashed")
    
    # Create Pyvis visualization
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )
    
    # Import from NetworkX
    net.from_nx(G)
    
    # Configure physics for better layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 200
        },
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)
    
    return G, net


def build_full_corpus_graph(all_records, include_topics=True, include_authors=True, 
                           include_cited_papers=False, include_cited_authors=False, 
                           min_topic_confidence=0.0):
    """
    Build a complete knowledge graph from all papers in the corpus.
    
    Shows the full network including:
    - All papers as nodes
    - All authors (with connections to their papers, revealing collaborations)
    - All topics (showing thematic clusters)
    - Cited papers and cited authors
    
    Parameters
    ----------
    all_records : list of dict
        All paper records from corpus_table.with_topics.jsonl
    include_topics : bool
        Add topic nodes connected to papers
    include_authors : bool
        Add author nodes connected to papers (reveals collaboration networks)
    include_cited_papers : bool
        Add cited paper nodes
    include_cited_authors : bool
        Add cited author nodes
    min_topic_confidence : float
        Minimum confidence score to include a topic (0.0 to 1.0)
    
    Returns
    -------
    tuple (networkx.DiGraph, pyvis.network.Network)
        - nx_graph: NetworkX directed graph
        - net: Pyvis Network ready for visualization
    """
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError(
            "Please install required packages: pip install networkx pyvis"
        ) from e
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Build DOI to record lookup for citations
    doi_to_record = {}
    for rec in all_records:
        rec_doi = (rec.get("DOI", "") or rec.get("doi", "")).strip().lower()
        if rec_doi:
            doi_to_record[rec_doi] = rec
    
    # Track all unique authors, topics, and categories across corpus
    all_authors = {}     # author_name -> list of node_ids
    all_topics = {}      # topic_label -> list of (node_id, confidence, category)
    all_categories = {}  # category_name -> list of node_ids (papers in that category)
    
    # First pass: Add all papers and collect authors/topics
    for rec in all_records:
        paper_id = rec.get("paper_id", "unknown")
        title = rec.get("title", "Untitled")
        n_citations = rec.get("n_citations", 0)
        doi = (rec.get("DOI", "") or rec.get("doi", "")).strip().lower()
        
        # Use DOI as node ID if available, otherwise use paper_id
        node_id = f"doi:{doi}" if doi else paper_id
        
        # Add paper node (size by citation count)
        G.add_node(
            node_id,
            label=paper_id,  # Show paper_id as label
            title=f"{title}\nDOI: {doi}" if doi else title,  # Full title on hover
            node_type="paper",
            color="#4A90E2",  # Blue for papers
            size=15 + min(n_citations * 3, 30),  # Size by citations
            shape="dot"
        )
        
        # Collect authors
        if include_authors:
            authors_raw = rec.get("authors", [])
            if isinstance(authors_raw, str):
                authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
            elif isinstance(authors_raw, list):
                authors = authors_raw
            else:
                authors = []
            
            for author in authors:
                author_name = author if isinstance(author, str) else author.get("name", "Unknown")
                if author_name not in all_authors:
                    all_authors[author_name] = []
                all_authors[author_name].append(node_id)
        
        # Collect topics (both categories and specific topics)
        if include_topics:
            topics = rec.get("topics", [])
            for topic in topics:
                category = topic.get("category", "General")
                topic_label = topic.get("label", "unknown topic")
                confidence = topic.get("confidence", 0.5)
                
                if confidence >= min_topic_confidence:
                    # Track categories
                    if category not in all_categories:
                        all_categories[category] = []
                    all_categories[category].append(node_id)
                    
                    # Track specific topics
                    if topic_label not in all_topics:
                        all_topics[topic_label] = []
                    all_topics[topic_label].append((node_id, confidence, category))
    
    # Second pass: Add author nodes and edges
    if include_authors:
        for author_name, node_ids in all_authors.items():
            # Author node size by number of papers
            author_id = f"author:{author_name}"
            G.add_node(
                author_id,
                label=author_name,
                title=f"Author: {author_name}\nPapers: {len(node_ids)}",
                node_type="author",
                color="#50C878",  # Green for authors
                size=10 + len(node_ids) * 5,  # Size by productivity
                shape="triangle"
            )
            
            # Connect author to all their papers
            for node_id in node_ids:
                G.add_edge(author_id, node_id, label="authored")
    
    # Third pass: Add category nodes (broad topics) and topic nodes (specific topics)
    if include_topics:
        # First add category nodes
        for category_name, paper_ids in all_categories.items():
            category_id = f"category:{category_name}"
            G.add_node(
                category_id,
                label=category_name,
                title=f"Category: {category_name}\nPapers: {len(paper_ids)}",
                node_type="category",
                color="#9B59B6",  # Purple for categories
                size=20 + len(paper_ids) * 4,  # Larger size for categories
                shape="box"
            )
            
            # Connect category to papers in it
            for paper_id in paper_ids:
                G.add_edge(paper_id, category_id, label="in_category")
        
        # Then add specific topic nodes
        for topic_label, topic_data in all_topics.items():
            topic_id = f"topic:{topic_label}"
            avg_confidence = sum(conf for _, conf, _ in topic_data) / len(topic_data)
            
            # Get category (should be same for all instances of this topic)
            category = topic_data[0][2] if topic_data else "General"
            category_id = f"category:{category}"
            
            G.add_node(
                topic_id,
                label=topic_label,
                title=f"Topic: {topic_label}\nCategory: {category}\nPapers: {len(topic_data)}\nAvg confidence: {avg_confidence:.2f}",
                node_type="topic",
                color="#FF6B6B",  # Red for specific topics
                size=10 + len(topic_data) * 2,  # Size by prevalence
                shape="ellipse"
            )
            
            # Connect topic to papers
            for paper_id, confidence, _ in topic_data:
                G.add_edge(paper_id, topic_id, label="has_topic", weight=confidence)
            
            # Connect topic to its category
            if category_id in G.nodes:
                G.add_edge(topic_id, category_id, label="belongs_to", style="dashed")
    
    # Fourth pass: Add cited papers
    if include_cited_papers:
        # Build title-to-record lookup for matching cited papers to corpus
        title_to_record = {}
        for rec in all_records:
            rec_title = rec.get("title", "").strip().lower()
            if rec_title:
                title_to_record[rec_title] = rec
        
        # Collect all cited authors across corpus
        all_cited_authors = {}  # {author_name: [citing_paper_ids]}
        
        for rec in all_records:
            rec_doi = (rec.get("DOI", "") or rec.get("doi", "")).strip().lower()
            citing_node_id = f"doi:{rec_doi}" if rec_doi else rec.get("paper_id", "unknown")
            
            # Handle cited_titles as either string (semicolon-separated) or list
            cited_titles_raw = rec.get("cited_titles", "")
            if isinstance(cited_titles_raw, str):
                cited_titles = [title.strip() for title in cited_titles_raw.split(";") if title.strip()]
            else:
                cited_titles = cited_titles_raw if cited_titles_raw else []
            
            # Get original citations array to access DOIs (it's stored as JSON string)
            citations_raw = rec.get("citations", "")
            citations = []
            if isinstance(citations_raw, str) and citations_raw:
                try:
                    citations = json.loads(citations_raw)
                except:
                    citations = []
            elif isinstance(citations_raw, list):
                citations = citations_raw
            
            # Build title-to-doi lookup from citations
            title_to_doi = {}
            for cit in citations:
                cit_title = cit.get("title", "").strip()
                cit_doi = (cit.get("DOI", "") or cit.get("doi", "")).strip()
                if cit_title:
                    title_to_doi[cit_title] = cit_doi
            
            # Add cited papers (whether in corpus or not)
            for cited_title in cited_titles:
                cited_title_clean = cited_title.strip()
                cited_title_lower = cited_title_clean.lower()
                
                # Get DOI if available
                cited_doi = title_to_doi.get(cited_title, "")
                
                # Check if cited paper is in our corpus (by title)
                if cited_title_lower in title_to_record:
                    # Paper IS in corpus - get its node_id
                    cited_rec = title_to_record[cited_title_lower]
                    corpus_doi = (cited_rec.get("DOI", "") or cited_rec.get("doi", "")).strip().lower()
                    cited_corpus_node_id = f"doi:{corpus_doi}" if corpus_doi else cited_rec.get("paper_id", "unknown")
                    # Add edge to existing corpus node
                    if cited_corpus_node_id in G.nodes:
                        G.add_edge(citing_node_id, cited_corpus_node_id, label="cites", color="#999999")
                else:
                    # Paper NOT in corpus - add as external node
                    cited_node_id = f"paper:{cited_title_clean[:50]}"  # Use title prefix as node ID
                    
                    if cited_node_id not in G.nodes:
                        # Truncate long titles for label
                        node_label = cited_title_clean[:25] + "..." if len(cited_title_clean) > 25 else cited_title_clean
                        node_title_text = f"{cited_title_clean}\n(not in corpus)"
                        if cited_doi:
                            node_title_text += f"\nDOI: {cited_doi}"
                        
                        G.add_node(
                            cited_node_id,
                            label=node_label,
                            title=node_title_text,
                            node_type="cited_paper",
                            color="#FFA500",  # Orange for external papers
                            size=10,
                            shape="dot"
                        )
                    # Add citation edge
                    G.add_edge(citing_node_id, cited_node_id, label="cites", color="#999999")
    
    # Fifth pass: Add cited authors
    if include_cited_authors:
        all_cited_authors = {}  # {author_name: [citing_paper_ids]}
        
        for rec in all_records:
            rec_doi = (rec.get("DOI", "") or rec.get("doi", "")).strip().lower()
            citing_node_id = f"doi:{rec_doi}" if rec_doi else rec.get("paper_id", "unknown")
            
            # Collect cited authors
            cited_authors_raw = rec.get("cited_authors", "")
            if isinstance(cited_authors_raw, str):
                cited_authors = [author.strip() for author in cited_authors_raw.split(";") if author.strip()]
            else:
                cited_authors = cited_authors_raw if cited_authors_raw else []
            
            for author_name in cited_authors:
                if author_name not in all_cited_authors:
                    all_cited_authors[author_name] = []
                all_cited_authors[author_name].append(citing_node_id)
        
        # Add cited author nodes
        for author_name, citing_papers in all_cited_authors.items():
            cited_author_id = f"cited_author:{author_name}"
            
            # Add cited author node
            if cited_author_id not in G.nodes:
                G.add_node(
                    cited_author_id,
                    label=author_name,
                    title=f"Cited Author: {author_name}\nCited by {len(citing_papers)} papers",
                    node_type="cited_author",
                    color="#FFD700",  # Gold for cited authors
                    size=8 + len(citing_papers) * 2,  # Size by citation count
                    shape="triangle"
                )
            
            # Connect cited author to papers that cite their work
            for citing_paper_id in citing_papers:
                G.add_edge(citing_paper_id, cited_author_id, label="cites_work_by", color="#CCCCCC", style="dashed")
    
    # Create Pyvis visualization
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )
    
    # Import from NetworkX
    net.from_nx(G)
    
    # Configure physics for larger graph
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 300
        },
        "barnesHut": {
          "gravitationalConstant": -15000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.02,
          "damping": 0.5
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)
    
    return G, net