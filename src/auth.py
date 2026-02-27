import streamlit as st
import os
import hmac  
import streamlit.components.v1 as components

def check_token():
    """
    Validates the user's session. 
    If not authenticated, displays a secure login form and halts execution.
    """
    expected_token = os.environ.get("TOKEN")

    
    if not expected_token:
        st.error("⚠️ Security Error: TOKEN environment variable is not set. Please contact support.")
        st.stop()

    # If the user is already authenticated in this specific browser session, let them pass
    if st.session_state.get("authenticated", False):
        return

    # --- Secure Login UI ---
    _, col, _ = st.columns([1, 2, 1])
    
    with col:
        st.markdown("<h2 style='text-align: center;'>🔒 Access Restricted</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>Please enter the secure password for this session.<br>"
            "<span style='font-size: 0.8em; color: gray;'>You can find this password on your Open OnDemand dashboard.</span></p>", 
            unsafe_allow_html=True
        )
        
        with st.form("login_form"):
            password_input = st.text_input(
                "Session Password", 
                type="password", 
                placeholder="Paste password here..."
            )
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                # SECURE FIX: hmac.compare_digest prevents timing attacks
                if hmac.compare_digest(password_input, expected_token):
                    st.session_state["authenticated"] = True
                    st.rerun()  # Reload the page as authenticated
                    user_token = st.context.cookies.get("textlab_auth_token", "")
                    expected_token = os.environ.get("TOKEN")

                    if expected_token != user_token:
                        components.html("""
                                <script>
                                    var token = new URLSearchParams(window.parent.location.search).get('token');
                                    if (token && !document.cookie.includes(token)) {
                                    window.parent.document.cookie = 'textlab_auth_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
                                    var expires = new Date(Date.now() + 30*864e5).toUTCString();
                                    window.parent.document.cookie = 'textlab_auth_token=' + token + '; expires=' + expires + '; path=/; secure; samesite=strict';
                                    window.parent.location.reload();
                                        }
                                </script>
                                        """, height=0)
                else:
                    st.error("❌ Invalid password. Please try again.")
    
    # Halt all further execution of the app
    st.stop()
