import streamlit as st
import os
from auth import initialize_cookies, check_token

st.set_page_config(page_title="TEXT LAB", layout="wide")

def main():
    cookies = initialize_cookies()
    check_token(cookies)

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
    static_folder = os.path.join(current_dir, "static")
    dsl_icon_path = os.path.join(static_folder, "dsl_icon.png")
    digiki_icon_path = os.path.join(static_folder, "digiki_icon.png")

    # Check if the files exist to avoid errors
    if not os.path.exists(dsl_icon_path) or not os.path.exists(digiki_icon_path):
        st.error("Static files not found. Please ensure dsl_icon.png and digiki_icon.png are in the 'static' folder.")
        return

    # Display images using Streamlit's native image handling
    st.image([dsl_icon_path, digiki_icon_path], width=150, caption=["DSL Icon", "Digiki Icon"])

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
        For more details on how to use Text Lab, check out our [GitHub README](https://github.com/ahmad-zurih/Text_lab).
        """
    )

if __name__ == "__main__":
    main()
