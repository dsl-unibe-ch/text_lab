"""Text Lab's web UI: FastAPI, Jinja2 templates and htmx.

It replaces the Streamlit UI (``textlab.ui.streamlit``) page by page; see
``docs/dev/web-ui-plan.md``. Until the switch, the launcher
(:mod:`textlab.ui.launch`) starts it only when ``TEXT_LAB_UI=web``.

- :mod:`~textlab.ui.web.app`: the app factory, middleware, error pages.
- :mod:`~textlab.ui.web.auth`: login with the session password, the session
  cookie, CSRF tokens.
- :mod:`~textlab.ui.web.jobs`: background jobs, one at a time, with live
  progress over server-sent events.
- :mod:`~textlab.ui.web.uploads`: receiving uploads.
- :mod:`~textlab.ui.web.render`: templates, icons, links.
- :mod:`~textlab.ui.web.pages`: the pages and the navigation.

See ``README.md`` in this folder for how a page is built.
"""
