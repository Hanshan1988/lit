"""LIT UI layout definitions for the sequence salience demo.

Three layouts are provided:

  * SALIENCE_ONLY  – minimal: data-table on the left, salience panel on the right.
  * SIMPLE         – simplified two-row layout (good for notebooks / demos).
  * FULL           – three-panel layout with embeddings and metrics.

Use the ``--layout`` CLI flag in server.py to select between them.
"""

from lit_nlp.api import layout

_m = layout.LitModuleName

# ---------------------------------------------------------------------------
# Layout definitions
# ---------------------------------------------------------------------------

SALIENCE_ONLY = layout.LitCanonicalLayout(
    left={
        "Examples": [_m.DataTableModule],
        "Editor": [_m.SingleDatapointEditorModule],
    },
    upper={
        "Salience": [_m.SequenceSalienceModule],
    },
    layoutSettings=layout.LayoutSettings(leftWidth=40),
    description="Minimal layout: data table + sequence salience.",
)

SIMPLE = layout.LitCanonicalLayout(
    upper={
        "Examples": [_m.SimpleDataTableModule],
        "Editor": [_m.SimpleDatapointEditorModule],
    },
    lower={
        "Salience": [_m.SequenceSalienceModule],
    },
    layoutSettings=layout.LayoutSettings(
        hideToolbar=True,
        mainHeight=40,
        centerPage=True,
    ),
    description="Simple top/bottom layout for sequence salience.",
)

FULL = layout.LitCanonicalLayout(
    left={
        "Data Table": [_m.DataTableModule],
        "Embeddings": [_m.EmbeddingsModule],
    },
    upper={
        "Datapoint Editor": [_m.SingleDatapointEditorModule],
        "Generators": [_m.GeneratorModule],
    },
    lower={
        "Salience": [_m.SequenceSalienceModule],
        "Metrics": [_m.MetricsModule],
    },
    layoutSettings=layout.LayoutSettings(mainHeight=40, leftWidth=40),
    description="Full three-panel layout with embeddings and metrics.",
)

# Exported mapping used by server.py
LAYOUTS = {
    "salience_only": SALIENCE_ONLY,
    "simple": SIMPLE,
    "full": FULL,
}

DEFAULT_LAYOUT = "salience_only"
