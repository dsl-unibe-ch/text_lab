import streamlit as st

def main():
    st.title("TEXT LAB")
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
        For more details on how to use Text Lab, check out our [GitHub
        README](https://github.com/ahmad-zurih/Text_lab).
        """
    )

if __name__ == "__main__":
    main()
