import streamlit as st
import ollama
import sys
import os
import datetime
from ollama import ResponseError
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(src_dir, "assets", "text_lab_logo.png")

favicon = Image.open(favicon_path)

st.set_page_config(
    page_title="Ollama Chat Interface",
    page_icon=favicon,
    layout="centered",
    initial_sidebar_state="expanded"
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token
from core.chat_engine import (
    check_ollama_server,
    get_gpu_name,
    is_model_loaded,
    extract_model_name,
    process_uploaded_files,
    get_response_generator,
    format_chat_history,
    estimate_tokens,
    chunk_text,
    get_chunk_answer,
    get_synthesis_generator,
    MAX_CONTEXT_TOKENS,
)

from core.model_config import get_available_models, is_high_memory_gpu

check_token()

def main():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.markdown(
        """
        <style>
            .main { max-width: 800px; margin: 0 auto; }
            [data-testid="stChatMessage"] { border: 1px solid #3f3f3f; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
            [data-testid="stChatMessage"]:has(div:has-text("User:")) { background: #313131; }
            [data-testid="stChatMessage"]:has(div:has-text("Assistant:")) { background: #1e1e1e; }
            .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # UI check for server status
    if not check_ollama_server():
        st.error("Could not connect to Ollama server.")
        st.info("Please check the log file: text_lab/ollama_server.log")
        st.stop()

    # --- GPU Detection & Model Filtering ---
    current_gpu = get_gpu_name()
    available_models_in_ui = get_available_models(current_gpu)

    if is_high_memory_gpu(current_gpu):
        gpu_badge = f"**High-Performance Mode** detected ({current_gpu})"
    else:
        gpu_badge = f" **Standard Mode** detected ({current_gpu}). Large models are hidden."

    if not available_models_in_ui:
        st.error("No models are configured. Please check src/config/models.json.")
        st.stop()

    # Sidebar
    st.sidebar.title("Model Selection")
    st.sidebar.info(gpu_badge)

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = available_models_in_ui[0]
    if st.session_state["selected_model"] not in available_models_in_ui:
        st.session_state["selected_model"] = available_models_in_ui[0]

    st.session_state["selected_model"] = st.sidebar.selectbox(
        "Select a model:",
        options=available_models_in_ui,
        index=available_models_in_ui.index(st.session_state["selected_model"])
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Context")
    
    # --- File Uploader Widget ---
    uploaded_files = st.sidebar.file_uploader(
        "Attach files (Max 4)", 
        type=["pdf", "txt", "csv", "xlsx"], 
        accept_multiple_files=True
    )
    
    # Enforce file count limit
    if uploaded_files and len(uploaded_files) > 4:
        st.sidebar.error("Maximum 4 files allowed. Please remove some.")
        uploaded_files = uploaded_files[:4]

    if st.sidebar.button("🗑️ Start New Chat"):
        st.session_state["messages"] = []
        st.rerun()


    st.sidebar.markdown(
        """
        ---
        ⚠️ **Disclaimer**
        The selected AI models may produce inaccurate, misleading, or inappropriate responses.
        """,
        unsafe_allow_html=True
    )

    model_name = st.session_state["selected_model"]

    # Pull logic...
    try:
        models_dict = ollama.list()
        models_list = models_dict.get("models", []) if isinstance(models_dict, dict) else getattr(models_dict, 'models', [])
        local_model_names = [extract_model_name(m) for m in models_list]
    except Exception as e:
        st.error(f"Error listing locally available models: {str(e)}")
        local_model_names = []

    if model_name not in local_model_names:
        st.write("\n\n")
        st.info(f"Model '{model_name}' not found locally. Pulling it now...")
        try:
            ollama.pull(model=model_name)
            st.success(f"Successfully pulled '{model_name}'.")
        except Exception as e:
            st.error(f"Error pulling model '{model_name}': {str(e)}")

    st.title("Ollama Chat Interface")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_text = st.chat_input("Type your message...")

    if user_text:
        # 1. Process files if they exist
        context_text = ""
        if uploaded_files:
            with st.spinner("Processing files..."):
                context_text, warnings = process_uploaded_files(uploaded_files)
                for warning in warnings:
                    st.warning(warning)
        
        # 2. Construct final message content
        if context_text:
            full_prompt = f"{context_text}\n\nUser Question: {user_text}"
            display_text = f"**[Uploaded {len(uploaded_files)} file(s)]**\n\n{user_text}"
        else:
            full_prompt = user_text
            display_text = user_text

        # 3. Add user message to history
        st.session_state["messages"].append({"role": "user", "content": display_text})
        with st.chat_message("user"):
            st.markdown(display_text)

        # 4. Generate response
        last_msg_obj = st.session_state["messages"][-1]
        original_content = last_msg_obj["content"]

        # --- DYNAMIC SPINNER LOGIC ---
        if is_model_loaded(model_name):
            spinner_text = "Thinking..."
        else:
            spinner_text = f"🚀 Loading **{model_name}** into GPU memory... This first run may take 1-2 minutes."

        needs_chunking = bool(context_text) and estimate_tokens(context_text) > MAX_CONTEXT_TOKENS

        try:
            if needs_chunking:
                chunks = chunk_text(context_text)
                partial_answers = []
                progress_placeholder = st.empty()

                for i, chunk_content in enumerate(chunks, 1):
                    progress_placeholder.info(
                        f"📄 Analyzing document part {i} of {len(chunks)} "
                        f"(~{estimate_tokens(context_text):,} tokens total)..."
                    )
                    answer = get_chunk_answer(
                        model_name, chunk_content, i, len(chunks),
                        user_text, st.session_state["messages"][:-1]
                    )
                    partial_answers.append(answer)

                progress_placeholder.info(f"🔗 Synthesizing responses from {len(chunks)} chunks...")
                with st.chat_message("assistant"):
                    synthesis_stream = get_synthesis_generator(
                        model_name, partial_answers,
                        user_text, st.session_state["messages"][:-1]
                    )
                    assistant_reply = st.write_stream(synthesis_stream)
                progress_placeholder.empty()
                st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
            else:
                last_msg_obj["content"] = full_prompt
                with st.spinner(spinner_text):
                    response_stream = get_response_generator(model_name, st.session_state["messages"])
                    with st.chat_message("assistant"):
                        assistant_reply = st.write_stream(response_stream)
                    st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
        except ResponseError as e:
            status = getattr(e, "status_code", "?")
            st.error(f"Ollama ResponseError (status={status})")
            st.code(str(e))
        finally:
            # Always restore display text so chat history shows the friendly version
            last_msg_obj["content"] = original_content

    if len(st.session_state["messages"]) > 0:
        chat_export = format_chat_history(st.session_state["messages"])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="📥 Download Conversation",
            data=chat_export,
            file_name=f"text_lab_chat_{timestamp}.md",
            mime="text/markdown"
        )
if __name__ == "__main__":
    main()
