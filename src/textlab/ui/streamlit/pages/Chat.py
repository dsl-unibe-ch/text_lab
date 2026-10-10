"""Chat page: a private chat with local models about the user's files.

Messages are answered through :mod:`textlab.features.chat.service`; with a
table attached, data questions go to the Visualization feature's agents,
which run in the background while the page polls them.
"""

import datetime
import os
import time

import streamlit as st
from ollama import ResponseError
from PIL import Image

from textlab.common import gpu_manager
from textlab.common.gpu_manager import get_gpu_name
from textlab.common.model_config import (
    get_available_models,
    is_high_memory_gpu,
)
from textlab.common.ollama import check_ollama_server
from textlab.features.chat import service
from textlab.ui.streamlit.auth import check_token
from textlab.ui.streamlit.components.gpu import free_gpu_for

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(app_dir, "assets", "text_lab_logo.png")
favicon = Image.open(favicon_path)

st.set_page_config(
    page_title="Ollama Chat Interface",
    page_icon=favicon,
    layout="centered",
    initial_sidebar_state="expanded",
)

check_token()

#: Most files a message can carry.
MAX_FILES = 4

#: Session-state keys of the attached table and a running analysis.
DATA_KEYS = (
    "chat_data_file_id",
    "chat_data_path",
    "chat_data_name",
    "chat_data_schema",
    "chat_tool_run",
)

STYLE = """
<style>
    .main { max-width: 800px; margin: 0 auto; }
    [data-testid="stChatMessage"] {
        border: 1px solid #3f3f3f; padding: 1rem; border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    [data-testid="stChatMessage"]:has(div:has-text("User:")) {
        background: #313131;
    }
    [data-testid="stChatMessage"]:has(div:has-text("Assistant:")) {
        background: #1e1e1e;
    }
    .block-container { padding-top: 1rem; }
</style>
"""


def _conversation_folder() -> str:
    """Return this conversation's folder for the attached table."""
    folder = st.session_state.get("chat_data_dir")
    if not folder or not os.path.isdir(folder):
        folder = service.conversation_folder()
        st.session_state["chat_data_dir"] = folder
    return folder


def _ensure_data_file(uploaded_files):
    """Save the first attached table, so the analysis agents can read it.

    Returns:
        The saved file's path, its name and a summary of its columns, or
        ``(None, None, None)`` without a table.
    """
    tabular = next(
        (f for f in uploaded_files or [] if service.is_table(f.name)), None
    )
    if tabular is None:
        return None, None, None

    file_id = (tabular.name, tabular.size)
    if st.session_state.get(
        "chat_data_file_id"
    ) == file_id and st.session_state.get("chat_data_path"):
        return (
            st.session_state["chat_data_path"],
            st.session_state["chat_data_name"],
            st.session_state.get("chat_data_schema", ""),
        )

    path, schema = service.prepare_data_file(
        tabular.name, tabular.getvalue(), _conversation_folder()
    )
    st.session_state["chat_data_file_id"] = file_id
    st.session_state["chat_data_path"] = path
    st.session_state["chat_data_name"] = tabular.name
    st.session_state["chat_data_schema"] = schema
    return path, tabular.name, schema


def _render_analysis_payload(payload: dict, run_id: str) -> None:
    """Show an assistant analysis turn: charts, then statistical results."""
    for idx, artifact in enumerate(payload.get("artifacts", [])):
        label = service.tool_label(artifact.get("tool_name", ""))
        if label:
            st.markdown(f"**{label}**")
        figure = service.render_figure(artifact)
        if figure is not None:
            st.plotly_chart(
                figure,
                use_container_width=True,
                key=f"chatplot_{run_id}_{idx}",
            )
        else:
            st.image(artifact["bytes"], caption=artifact["filename"])
        if artifact.get("code"):
            with st.expander(
                f"View Source Code: {label or artifact['filename']}"
            ):
                st.code(artifact["code"], language="python")

    for item in payload.get("stats", []):
        with st.expander(
            item.get("title", "Statistical Result"), expanded=False
        ):
            st.markdown(item.get("result", ""))
            if item.get("code"):
                st.code(item["code"], language="python")


def _render_tool_run_section() -> None:
    """Poll the running analysis, show its activity, and finish the turn.

    When the run ends, the assistant's answer is added to the conversation.
    """
    run = st.session_state["chat_tool_run"]
    is_complete = not run.running

    with st.chat_message("assistant"):
        with st.status(
            "Analysing your data...",
            expanded=not is_complete,
            state="complete" if is_complete else "running",
        ):
            logs = list(run.logs)
            for log_type, msg in logs:
                if log_type == "error":
                    st.error(msg)
                elif log_type == "warning":
                    st.warning(msg)
                else:
                    st.write(msg)
            if not logs:
                st.caption("Starting agents...")

    if run.running:
        time.sleep(1)
        st.rerun()
        return

    messages = st.session_state["messages"]
    if run.status == "done":
        messages.append(
            {
                "role": "assistant",
                "content": run.result.summary or "Analysis complete.",
                "analysis": run.result.to_payload(),
            }
        )
    elif run.status == "timeout":
        minutes = service.ANALYSIS_TIMEOUT_SECONDS // 60
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"The analysis exceeded the {minutes}-minute limit. Try "
                    "a simpler request or a smaller dataset."
                ),
            }
        )
    elif run.status == "cancelled":
        messages.append(
            {"role": "assistant", "content": "Analysis was cancelled."}
        )
    else:
        messages.append(
            {
                "role": "assistant",
                "content": (
                    "An error occurred while processing your request: "
                    f"{run.error or ''}"
                ),
            }
        )

    del st.session_state["chat_tool_run"]
    st.rerun()


def _render_sidebar(available_models, gpu_badge):
    """Draw the model choice, attachments and conversation controls.

    Returns:
        The selected model and the attached files.
    """
    st.sidebar.title("Model Selection")
    st.sidebar.info(gpu_badge)

    if st.session_state.get("selected_model") not in available_models:
        st.session_state["selected_model"] = available_models[0]
    st.session_state["selected_model"] = st.sidebar.selectbox(
        "Select a model:",
        options=available_models,
        index=available_models.index(st.session_state["selected_model"]),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Context")
    uploaded_files = st.sidebar.file_uploader(
        f"Attach files (Max {MAX_FILES})",
        type=["pdf", "txt", "csv", "tsv", "xls", "xlsx", "json"],
        accept_multiple_files=True,
    )
    if uploaded_files and len(uploaded_files) > MAX_FILES:
        st.sidebar.error(
            f"Maximum {MAX_FILES} files allowed. Please remove some."
        )
        uploaded_files = uploaded_files[:MAX_FILES]
    return st.session_state["selected_model"], uploaded_files


def _start_new_chat():
    """Forget the conversation and remove its attached table and charts."""
    st.session_state["messages"] = []
    folder = st.session_state.pop("chat_data_dir", None)
    if folder:
        service.discard_conversation_folder(folder)
    for key in DATA_KEYS:
        st.session_state.pop(key, None)


def _ensure_model(model_name):
    """Pull the selected model if the Ollama server does not have it."""
    if service.model_installed(model_name):
        return
    st.write("\n\n")
    st.info(f"Model '{model_name}' not found locally. Pulling it now...")
    try:
        service.pull_model(model_name)
        st.success(f"Successfully pulled '{model_name}'.")
    except Exception as e:
        st.error(f"Error pulling model '{model_name}': {e}")


def _answer(model_name, user_text, uploaded_files):
    """Answer a message with the chat model, streaming the reply."""
    context_text = ""
    if uploaded_files:
        with st.spinner("Processing files..."):
            context_text, warnings = service.read_documents(
                [(f.name, f.getvalue()) for f in uploaded_files]
            )
            for warning in warnings:
                st.warning(warning)

    if context_text:
        display_text = (
            f"**[Uploaded {len(uploaded_files)} file(s)]**\n\n{user_text}"
        )
    else:
        display_text = user_text

    history = list(st.session_state["messages"])
    st.session_state["messages"].append(
        {"role": "user", "content": display_text}
    )
    with st.chat_message("user"):
        st.markdown(display_text)

    if service.is_model_loaded(model_name):
        spinner_text = "Thinking..."
    else:
        spinner_text = (
            f"Loading **{model_name}** into GPU memory... This first run may "
            "take 1-2 minutes."
        )

    progress = st.empty()
    try:
        with st.spinner(spinner_text):
            stream = service.answer(
                model_name,
                history,
                user_text,
                context_text,
                on_progress=lambda update: progress.info(update.message),
            )
            # The stream is lazy: take the first piece under the spinner, so
            # it shows until the model is loaded and starts answering.
            first_chunk = next(stream, "")

        def _reply():
            if first_chunk:
                yield first_chunk
            yield from stream

        with st.chat_message("assistant"):
            assistant_reply = st.write_stream(_reply())
        st.session_state["messages"].append(
            {"role": "assistant", "content": assistant_reply}
        )
    except ResponseError as e:
        status = getattr(e, "status_code", "?")
        st.error(f"Ollama ResponseError (status={status})")
        st.code(str(e))
    finally:
        progress.empty()


def _render_downloads():
    """Offer the conversation as Markdown and, with charts, as HTML."""
    messages = st.session_state["messages"]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="Download Conversation (.md)",
        data=service.format_chat_history(messages),
        file_name=f"text_lab_chat_{timestamp}.md",
        mime="text/markdown",
    )
    if service.has_analysis_plots(messages):
        st.sidebar.download_button(
            label="Download with Plots (.html)",
            data=service.format_chat_history_html(messages),
            file_name=f"text_lab_chat_{timestamp}.html",
            mime="text/html",
            help=(
                "Includes interactive charts. Markdown export can't show "
                "interactive plots."
            ),
        )


def main():
    """Draw the page and answer the user's message."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.markdown(STYLE, unsafe_allow_html=True)

    if not check_ollama_server():
        st.error("Could not connect to Ollama server.")
        st.info("Please check the log file: text_lab/ollama_server.log")
        st.stop()

    current_gpu = get_gpu_name()
    available_models = get_available_models(current_gpu)
    if is_high_memory_gpu(current_gpu):
        gpu_badge = f"**High-Performance Mode** detected ({current_gpu})"
    else:
        gpu_badge = (
            f" **Standard Mode** detected ({current_gpu}). Large models are "
            "hidden."
        )
    if not available_models:
        st.error(
            "No models are configured. Please check src/config/models.json."
        )
        st.stop()

    model_name, uploaded_files = _render_sidebar(available_models, gpu_badge)

    # Save an attached table so the data-analysis agents can read it.
    data_file_path, data_file_name, data_schema = _ensure_data_file(
        uploaded_files
    )
    if data_file_path:
        st.sidebar.success(
            f"Data tools enabled for **{data_file_name}**. Ask for plots or "
            "statistics and I'll analyse it."
        )

    st.sidebar.markdown("---")
    if st.sidebar.button("Start New Chat"):
        _start_new_chat()
        st.rerun()

    st.sidebar.markdown(
        """
        ---
        **Disclaimer**
        The selected AI models may produce inaccurate, misleading, or
        inappropriate responses.
        """,
        unsafe_allow_html=True,
    )

    _ensure_model(model_name)

    st.title("Ollama Chat Interface")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("analysis"):
                _render_analysis_payload(
                    msg["analysis"], msg["analysis"].get("run_id", "hist")
                )

    # While an analysis runs, poll it instead of taking new input.
    if "chat_tool_run" in st.session_state:
        _render_tool_run_section()
        return

    user_text = st.chat_input("Type your message...")
    if user_text:
        # Chat keeps its LLM; other features' models are released so they
        # cannot starve the chat model of VRAM.
        free_gpu_for(gpu_manager.LLM, ollama_model=model_name)

    if user_text and data_file_path:
        with st.spinner("Deciding how to answer..."):
            use_tools, instruction = service.decide_tool_use(
                model_name,
                user_text,
                data_schema or "",
                chat_history=service.plain_messages(
                    st.session_state["messages"]
                ),
            )
        if use_tools:
            with st.chat_message("user"):
                st.markdown(user_text)
            st.session_state["messages"].append(
                {"role": "user", "content": user_text}
            )
            st.session_state["chat_tool_run"] = service.start_data_analysis(
                instruction or user_text, data_file_path, model_name
            )
            st.rerun()
            return

    if user_text:
        _answer(model_name, user_text, uploaded_files)

    if st.session_state["messages"]:
        _render_downloads()


if __name__ == "__main__":
    main()
