import streamlit as st
import os

def check_token():
    expected_token = os.environ.get("TOKEN")
    if expected_token is None:
        raise RuntimeError("TOKEN environment variable not set.")

    # Check if token already stored in session
    if "user_token" not in st.session_state:
        query_params = st.query_params
        token_in_url = query_params.get("token")

        if token_in_url:
            st.session_state["user_token"] = token_in_url
        else:
            st.error("❌ Access Denied: Missing token in URL.")
            st.stop()

    # Now validate the token
    if st.session_state["user_token"] != expected_token:
        st.error("❌ Access Denied: Invalid token.")
        st.stop()
