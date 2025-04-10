import os
import streamlit as st

def main():
    st.title("TEXT LAB")
    
    # Build absolute paths relative to the current file's directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dsl_icon_path = os.path.join(current_dir, "dsl_icon.png")
    digiki_icon_path = os.path.join(current_dir, "digiki_icon.png")
    
    # Check that the files exist (for debugging purposes)
    if not os.path.exists(dsl_icon_path):
        st.error(f"File not found: {dsl_icon_path}")
    if not os.path.exists(digiki_icon_path):
        st.error(f"File not found: {digiki_icon_path}")
    
    # Display logos side by side.
    col1, col2 = st.columns(2)
    with col1:
        st.image(dsl_icon_path, width=150)   # Adjust width as needed.
    with col2:
        st.image(digiki_icon_path, width=150)
    
    st.markdown(
        """
        **Welcome to Text Lab** – an interactive application that provides a range of
        Natural Language Processing (NLP) tools. Currently, you can:
        - **Transcribe** audio files using Whisper.
        - **Chat** with a basic AI chatbot.

        More NLP features and enhancements are on the way. This project is
        still under active development, so expect frequent updates and new
        capabilities soon!

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
