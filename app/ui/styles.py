"""Custom CSS styles for modern, minimal AI product aesthetics."""

from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
/* Modern fonts and layout smoothing */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Hide Streamlit default header decoration and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
@media (prefers-color-scheme: dark) {
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
}

/* Sidebar session buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease-in-out;
}

/* Suggestion chip buttons in empty state */
.suggestion-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 18px;
    background-color: #ffffff;
    cursor: pointer;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 10px;
}
.suggestion-card:hover {
    border-color: #94a3b8;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}
@media (prefers-color-scheme: dark) {
    .suggestion-card {
        background-color: #1e293b;
        border-color: #334155;
    }
    .suggestion-card:hover {
        border-color: #64748b;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
}

/* Document cards in sidebar */
.doc-card {
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    background-color: #ffffff;
    margin-bottom: 8px;
}
@media (prefers-color-scheme: dark) {
    .doc-card {
        background-color: #1e293b;
        border-color: #334155;
    }
}
.doc-card-title {
    font-weight: 600;
    font-size: 0.88rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-card-meta {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 2px;
}

/* Sources section & badges */
.source-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    background-color: #f1f5f9;
    color: #334155;
    margin-right: 6px;
}
@media (prefers-color-scheme: dark) {
    .source-badge {
        background-color: #334155;
        color: #e2e8f0;
    }
}

/* Message styling */
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

/* Chat input sticky footer */
div[data-testid="stChatInput"] {
    border-radius: 12px;
}
</style>
"""


def apply_custom_styles() -> None:
    """Inject custom styles into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
