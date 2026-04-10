# Other Features

##  AI Chat

Text Lab provides a secure Chat interface allowing you to interact with Large Language Models (LLMs) running locally on the University cluster. You can also upload documents (PDF, TXT, CSV, XLSX) with a limited size and token count to summarize, translate, and interact with them.

###  Data Privacy & Security (Zero-Footprint Architecture)

We understand that you may be uploading sensitive, unpublished, or proprietary research data to the Chat tool. Text Lab is designed with a **"Zero-Footprint"** architecture to ensure your data remains strictly confidential.

Here is exactly what happens to your data when you use the Chat feature, and the underlying technologies used:

* **Strictly In-Memory Processing:** When you upload documents to the chat, they are processed entirely in the server's temporary volatile memory (RAM). Your files are never explicitly saved, copied, or written to your university home directory.
    * *Technical Transparency:* The app uses Streamlit's `uploaded_file.getvalue()` to read the file into RAM. For PDFs, it uses PyMuPDF (`fitz.open(stream=...)`); for tabular data, it uses Pandas wrapped in a memory buffer (`pd.read_csv(BytesIO(...))`).
* **Ephemeral Sessions:** Your chat history and uploaded contexts are permanently tied to your active browser tab using Streamlit's `st.session_state`. The moment you click "Start New Chat", refresh the page, close the browser, or your HPC job ends, **your entire conversation and data are instantly and permanently destroyed** by Python's garbage collector.
* **No AI Training:** The AI models run locally on the UBELIX compute nodes via Ollama. They only read your data to answer your current question (inference). **The models do not learn from your data**, and your data is never used to train or improve the AI.
* **Network Isolation:** All data transfers happen internally within the University of Bern's secure HPC network. No data is ever sent to external APIs like OpenAI, Google, or Anthropic.

> ** Note on Large File Uploads and Temporary Storage:** > To protect the AI's context window, the Chat interface specifically restricts individual document uploads to **10MB**. However, the underlying Text Lab server is configured to accept much larger files (up to 10GB) to support the Data Visualisation and Knowledge Graph tools. 
> 
> If you attempt to upload an extremely large file to the Chat, the underlying web framework (Streamlit) may temporarily spool the file to the operating system's temporary directory (`$TMPDIR`) to prevent RAM overflow *before* the Chat logic rejects it. For the strictest data privacy, please adhere to the 10MB limit.

### Usage
Use this for summarizing sensitive text, brainstorming research ideas, querying your own custom datasets, or drafting emails safely within the university network.

---

##  Visualize Data

This tool allows you to upload datasets (CSV/Excel) and use LLMs to generate Python code for visualization.

* **How it works:** Describe the plot you want to see (e.g., *"Show a bar chart of sales by region"*). The system generates the code and renders the plot instantly.
* **Code Transparency:** The tool outputs the exact Python code it generated so you can verify, download, and reuse it in your own environment.
* **Architecture:** The visualization relies on an internal Model Context Protocol (MCP) system. It has a specific suite of standard plots and the ability to write custom Matplotlib/Seaborn code.

*Note: The primary purpose of this feature is rapid, exploratory data analysis rather than producing highly complex, publication-ready interactive dashboards.*