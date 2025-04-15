import streamlit as st

st.set_page_config(page_title="TEXT LAB", layout="wide")

import os
import base64

from auth import check_token

check_token()

def get_img_as_base64(file_path):
    """Read an image file and return its base64 encoded string."""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def main():
    st.title("TEXT LAB")
    
    # Custom CSS for styling the logo container.
    css = """
    <style>
    .logo-container {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 40px;
      margin-bottom: 20px;
    }
    .logo-container img {
      border: 2px solid #ddd;
      border-radius: 8px;
      box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.1);
      max-width: 150px;
      height: auto;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Get the absolute paths to the images.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dsl_icon_path = os.path.join(current_dir, "dsl_icon.png")
    digiki_icon_path = os.path.join(current_dir, "digiki_icon.png")

    # Encode images as base64 strings.
    dsl_base64 = get_img_as_base64(dsl_icon_path)
    digiki_base64 = get_img_as_base64(digiki_icon_path)

    # Create an HTML container to display the logos side by side.
    html_logo_container = f"""
    <div class="logo-container">
      <img src="data:image/png;base64,{dsl_base64}" alt="DSL Icon">
      <img src="data:image/png;base64,{digiki_base64}" alt="Digiki Icon">
    </div>
    """
    st.markdown(html_logo_container, unsafe_allow_html=True)

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
        Using the tools in Text lab insure that your data is only processed privately within the University Network and infrastructure. This may be a privacy requeirment in many cases.

        **Project details**:
        - **Maintained by**: The Data Science Lab (DSL)
        - **Funded by**: The Digitalisation Commission
        - **For questions or issues**: [support.dsl@unibe.ch](mailto:support.dsl@unibe.ch)

        **Documentation**:
        For more details on how to use Text Lab, check out our [GitHub README](https://github.com/ahmad-zurih/Text_lab).
        """
    )

if __name__ == "__main__":
    main()
