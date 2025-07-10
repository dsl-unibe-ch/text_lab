import streamlit as st
import os
import base64
import subprocess
from ollama import chat
from ollama import ChatResponse

st.set_page_config(page_title="TEXT LAB", layout="wide")

from auth import check_token

def get_img_as_base64(file_path):
    """Read an image file and return its base64 encoded string."""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def main():
    check_token()

    st.title("TEXT LAB")

    # Custom CSS for styling the logos.
    css = """
    <style>
        .main-logo {
          display: flex;
          justify-content: center;
          margin-bottom: 30px;
        }
        .main-logo img {
          max-width: 200px;
          border: none;
        }
        .sub-logos {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 40px;
          margin-top: 40px;
        }
        .sub-logos img {
          border: 2px solid #ddd;
          border-radius: 8px;
          box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.1);
          max-width: 120px;
          height: auto;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_logo_path = os.path.join(current_dir, "text_lab_logo.png")
    dsl_icon_path = os.path.join(current_dir, "dsl_icon.png")
    digiki_icon_path = os.path.join(current_dir, "digiki_icon.png")

    # Convert images to base64
    main_logo_base64 = get_img_as_base64(main_logo_path)
    dsl_base64 = get_img_as_base64(dsl_icon_path)
    digiki_base64 = get_img_as_base64(digiki_icon_path)

    # Render main logo
    st.markdown(f"""
    <div class="main-logo">
      <img src="data:image/png;base64,{main_logo_base64}" alt="Text Lab Logo">
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        **Welcome to Text Lab** – an interactive application that provides a range of
        Natural Language Processing (NLP) tools. Currently, you can:
        - **Transcribe** audio files using Whisper.
        - **Chat** with a basic AI chatbot.

        More NLP features and enhancements are on the way. This project is
        still under active development, so expect frequent updates and new
        capabilities soon!

        **Why Use Text Lab?**
        Using the tools in Text lab insure that your data is only processed privately within the University Network and infrastructure. This may be a privacy requirement in many cases.

        **Project details**:
        - **Maintained by**: The Data Science Lab (DSL)
        - **Funded by**: The Digitalisation Commission
        - **For questions or issues**: [support.dsl@unibe.ch](mailto:support.dsl@unibe.ch)

        **Documentation**:
        For more details on how to use Text Lab, check out our [GitHub README](https://github.com/dsl-unibe-ch/text_lab).
        """
    )

    # Render supporting logos at the bottom
    st.markdown(f"""
    <div class="sub-logos">
      <img src="data:image/png;base64,{dsl_base64}" alt="DSL Icon">
      <img src="data:image/png;base64,{digiki_base64}" alt="Digiki Icon">
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()