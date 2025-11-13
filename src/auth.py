import streamlit as st
import os
from streamlit_cookies_manager import EncryptedCookieManager

cookies = EncryptedCookieManager(
    prefix="textlab_",
    password=os.environ.get("TOKEN", "fallback_password_for_dev"),
)
if not cookies.ready():
    st.stop()


def check_token():
    expected_token = os.environ.get("TOKEN")
    if expected_token is None:
        raise ValueError("TOKEN environment variable not set.")

    # Automatically store token from URL into cookies
    token_from_url = st.query_params.get("token")
    if token_from_url:
        cookies["auth_token"] = token_from_url
        cookies.save()

    # Check token in cookies
    user_token = cookies.get("auth_token")

    if user_token is None:
        st.warning("This app requires authentication.")
        token_input = st.text_input("Enter access token", type="password")
        if token_input:
            cookies["auth_token"] = token_input
            cookies.save()
            st.rerun()
        st.stop()

    if user_token != expected_token:
        st.error("❌ Invalid token.")
        cookies["auth_token"] = ""
        cookies.save()
        st.stop()
