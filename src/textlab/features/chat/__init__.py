"""Backend for the Chat feature: a private chat about the user's files.

Interfaces use :mod:`.service`. The modules behind it: ``documents``
(reading attachments), ``generation`` (streamed answers), ``router`` (chat
or data analysis) and ``exports`` (Markdown and HTML). Data questions are
answered by the Visualization feature's agents.

See ``README.md`` in this folder for the pipeline and the files written.
"""
