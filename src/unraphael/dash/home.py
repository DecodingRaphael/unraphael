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
"""
)

# Center-align using Streamlit's layout
col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is wider

with col2:
    st.image(str(data_directory / 'raphael.jpg'))

st.markdown(
    """
### Pages

1. **[Preprocess images](preprocess)**
   - Clean and enhance your images
   - Apply filters, adjust parameters, and remove backgrounds
   - See also https://unraphael.readthedocs.io/en/latest/steps/preprocessing/

2. **[Image similarity](image_similarity)**
   - Compare images using advanced feature detection
   - Analyze structural similarities between paintings

3. **[Detect objects](detect_objects)**
   - Identify and segment figures in paintings
   - Extract specific elements for detailed analysis
   - See also https://unraphael.readthedocs.io/en/latest/steps/segmentation/

4. **[Align and compare](compare)**
   - Align and overlay multiple images
   - Analyze differences between similar paintings
   - See also https://unraphael.readthedocs.io/en/latest/steps/alignment/

5. **[Clustering images based on structural similarities](cluster)**
   - Group similar paintings automatically
   - Discover patterns across multiple artworks
   - See also https://unraphael.readthedocs.io/en/latest/steps/clustering/

6. **[Analyzing ratios](ratios)**
   - Compare size ratios between paintings and photographs
   - Analyze proportional relationships in artworks
   - See also https://unraphael.readthedocs.io/en/latest/steps/analysis/

### Credits

This project is maintained by the [Netherlands eScience Center](https://www.esciencecenter.nl/) in collaboration with the [Department of History and Art History](https://www.uu.nl/en/organisation/department-of-history-and-art-history) at the University of Utrecht.

**Principal Investigator:** Dr. L. Costiner ([l.costiner@uu.nl](mailto:l.costiner@uu.nl))
**Technical Support:** Thijs Vroegh, Stef Smeets ([t.vroegh@esciencecenter.nl](mailto:t.vroegh@esciencecenter.nl), [s.smeets@esciencecenter.nl](mailto:s.smeets@esciencecenter.nl))

Supported through a *Small-Scale Initiatives Digital Approaches to the Humanities* grant.

### More information

- [Source code](https://github.com/DecodingRaphael/unraphael)
- [Documentation](https://unraphael.readthedocs.io/)
- [Research Software Directory](https://research-software-directory.org/projects/renaissance)
""",
    unsafe_allow_html=True,
)
