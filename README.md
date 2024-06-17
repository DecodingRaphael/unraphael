[![Documentation Status](https://readthedocs.org/projects/unraphael/badge/?version=latest)](https://unraphael.readthedocs.io/en/latest/?badge=latest)
![Coverage](https://gist.githubusercontent.com/stefsmeets/808729a4ba7f123f650e32c499e143a4/raw/covbadge.svg)
[![Tests](https://github.com/DecodingRaphael/unraphael/actions/workflows/tests.yaml/badge.svg)](https://github.com/DecodingRaphael/unraphael/actions/workflows/tests.yaml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unraphael)](https://pypi.org/project/unraphael/)
[![PyPI](https://img.shields.io/pypi/v/unraphael.svg?style=flat)](https://pypi.org/project/unraphael/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11192044.svg)](https://doi.org/10.5281/zenodo.11192044)

![Unraphael banner](https://raw.githubusercontent.com/DecodingRaphael/unraphael/main/src/unraphael/data/logo.png#gh-light-mode-only)
![Unraphael banner](https://raw.githubusercontent.com/DecodingRaphael/unraphael/main/src/unraphael/data/logo-dark.png#gh-dark-mode-only)

# Unraphael

**Unraphael** is a digital workflow tool that uses computer vision to unravel the artistic practice of Raphael (Raffaello Sanzio, 1483-1520), while providing new digital approaches for the study of artistic practice in art history. Dozens of faithful reproductions survive of Raphael's paintings, attesting to the lucrative practice of serial production of paintings within the artist's workshop and to the lasting demand for the master's designs. This tool aims to provide new insights into Raphael's working methods through new digital approaches for the study of artistic practice in art history.

To install:

```console
pip install unraphael
```

## Try unraphael in your browser!

You can also [try unraphael directly from your browser](https://unraphael.streamlit.app/).

<table>
  <tr>
     <td><a href =https://unraphael.streamlit.app/image_similarity>Image similarity</a></td>
     <td><a href =https://unraphael.streamlit.app/image_similarity>Image preprocessing</a></td>
     <td><a href =https://unraphael.streamlit.app/image_similarity>Object detection</a></td>
     <td><a href =https://unraphael.streamlit.app/image_similarity>Image comparison</a></td>
  </tr>
  <tr>
    <td><img src="docs/_static/dash_image_sim.png" alt="Image similarity" width="85%"/></td>
    <td><img src="docs/_static/dash_preprocess.png" alt="Image preprocessing" width="85%"/></td>
    <td><img src="docs/_static/dash_detect.png" alt="Object detection" width="85%"/></td>
    <td><img src="docs/_static/dash_compare.png" alt="Image comparison" width="85%"/></td>
  </tr>
</table>

## Using the unraphael dashboard locally

To install and use the dashboard locally:

```console
pip install unraphael[dash]
unraphael-dash
```

## Development

Check out our [Contributing Guidelines](CONTRIBUTING.md#Getting-started-with-development) to get started with development.

Suggestions, improvements, and edits are most welcome.
