from __future__ import annotations

import streamlit as st

from unraphael.locations import data_directory


def set_custom_css(stylesheet: str = 'custom.css'):
    """Set custom css here."""
    add_sidebar_logo()

    fn = str(data_directory / stylesheet)

    with open(fn) as f:
        data = f.read()

    st.write(
        f'<style>\n{data}\n</style>',
        unsafe_allow_html=True,
    )


def add_sidebar_logo():
    """https://docs.streamlit.io/develop/api-reference/media/st.logo"""
    png_file = str(data_directory / 'logo-dark.png')
    st.logo(png_file, link='https://unraphael.readthedocs.org')
