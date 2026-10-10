import hmac
import os

import streamlit as st


def check_token():
    """Require the configured session token before rendering a page."""
    expected_token = os.environ.get("TOKEN")

    if not expected_token:
        st.error(
            "⚠️ Security Error: TOKEN environment variable is not set. "
            "Please contact support."
        )
        st.stop()

    if st.session_state.get("authenticated", False):
        return

    _, col, _ = st.columns([1, 2, 1])

    with col:
        st.markdown(
            "<h2 style='text-align: center;'>🔒 Access Restricted</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center;'>Please enter the secure password for this session.<br>"
            "<span style='font-size: 0.8em; color: gray;'>You can find this password "
            "on your Open OnDemand dashboard.</span></p>",
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            password_input = st.text_input(
                "Session Password",
                type="password",
                placeholder="Paste password here...",
            )
            submit_button = st.form_submit_button("Login", use_container_width=True)

            if submit_button:
                if hmac.compare_digest(password_input, expected_token):
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid password. Please try again.")
    
    st.stop()
