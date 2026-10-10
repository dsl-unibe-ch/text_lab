"""The Streamlit app.

``Home.py`` is the entry point that ``template/script.sh.erb`` starts;
Streamlit lists every script in ``pages/`` as a page in the sidebar.
``auth.py`` checks the session password, ``components/`` holds widgets
shared by several pages and ``assets/`` holds the images.

The app imports the ``textlab`` package, so ``src`` must be on
``PYTHONPATH``; the launch script sets it.
"""
