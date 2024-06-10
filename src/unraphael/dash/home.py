from __future__ import annotations

import streamlit as st
from styling import data_directory, set_custom_css

from unraphael import __version__

st.set_page_config(
    page_title='Unraphael dashboard',
    page_icon='🖼️',
    layout='centered',
    initial_sidebar_state='auto',
    menu_items={
        'Get Help': 'https://unraphael.readthedocs.io',
        'Report a bug': 'https://github.com/DedodingRaphael/unraphael/issues',
        'About': (
            f'**unraphael**: a dashboard for unraphael ({__version__}). '
            '\n\nPython toolkit for *unraveling* image similarity with a focus '
            'on artistic practice. '
            '\n\nFor more information, see: https://github.com/DedodingRaphael/unraphael'
        ),
    },
)

set_custom_css()

st.image(str(data_directory / 'logo-dark.png'))

st.markdown(
    """
Unraphael is a digital workflow tool that uses computer vision to unravel the artistic
practice of Raphael (Raffaello Sanzio, 1483-1520), while providing new digital approaches
for the study of artistic practice in art history. Dozens of faithful reproductions survive
of Raphael's paintings, attesting to the lucrative practice of serial production of
paintings within the artist's workshop and to the lasting demand for the master's designs.

This tool aims to provide new insights into Raphael's working methods through new digital
approaches for the study of artistic practice in art history.

### Pages

- <a href="/image_similarity" target="_parent">Image Similarity</a>
- <a href="/preprocess" target="_parent">Preprocess</a>
- <a href="/detect_objects" target="_parent">Detect objects</a>

### More information

- [Source code](https://github.com/DecodingRaphael/unraphael)
- [Documentation](https://unraphael.readthedocs.io/)
""",
    unsafe_allow_html=True,
)
