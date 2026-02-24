import streamlit as st
import os

def check_token():
    user_token = st.context.cookies.get("textlab_auth_token")

    expected_token = os.environ.get("TOKEN")

    if expected_token is None:
        st.stop()

        raise ValueError("TOKEN environment variable not set.")

    if user_token is None:
        st.warning(f"This app requires authentication.")

        st.stop()

    if user_token != expected_token:
        st.error(f"❌ Invalid token.")

        st.stop()
