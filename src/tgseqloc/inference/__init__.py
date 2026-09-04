"""Model inference stages that run inside the pipeline.

Everything here loads a model and produces one of the canonical formats in
``tgseqloc.data.formats``. Parsing artifacts that another tool already produced
belongs in ``tgseqloc.data`` instead: a source reads, a backend infers.
"""
