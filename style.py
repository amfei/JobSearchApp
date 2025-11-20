# =====================================================
# 🎨 Streamlit Custom Styles
# =====================================================

import streamlit as st

def apply_custom_style():
    """Inject professional CSS styles into Streamlit for a clean, modular UI."""

    st.markdown("""
    <style>
        /* ====== GLOBAL STYLING ====== */
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", sans-serif;
            color: #FAFAFA;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* ====== HEADINGS ====== */
        h1, h2, h3, h4 {
            color: #4FC3F7 !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }

        /* ====== BUTTONS ====== */
        .stButton>button {
            background: linear-gradient(135deg, #2196F3, #64B5F6);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.4em;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #42A5F5, #90CAF9);
            transform: scale(1.03);
        }

        /* ====== INPUTS ====== */
        .stTextInput>div>div>input,
        .stTextArea textarea {
            border-radius: 6px !important;
            border: 1px solid #2E3A46 !important;
            background-color: #1A1D25 !important;
            color: #FAFAFA !important;
        }

        /* ====== CARDS ====== */
        .card {
            background-color: #1A1D25;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        }

        /* ====== SECTION TITLES ====== */
        .section-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #90CAF9;
            margin-top: 1.2rem;
            margin-bottom: 0.6rem;
        }

        /* ====== LINKS ====== */
        a {
            color: #64B5F6 !important;
            text-decoration: none !important;
        }
        a:hover {
            text-decoration: underline !important;
            color: #90CAF9 !important;
        }

        /* ====== PROGRESS BAR ====== */
        .stProgress > div > div > div > div {
            background-color: #42A5F5;
        }

        /* ====== FOOTER ====== */
        footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)
