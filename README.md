[![Documentation Status](https://readthedocs.org/projects/unraphael/badge/?version=latest)](https://unraphael.readthedocs.io/en/latest/?badge=latest)
![Coverage](https://gist.githubusercontent.com/stefsmeets/808729a4ba7f123f650e32c499e143a4/raw/covbadge.svg)
[![Tests](https://github.com/DecodingRaphael/unraphael/actions/workflows/tests.yaml/badge.svg)](https://github.com/DecodingRaphael/unraphael/actions/workflows/tests.yaml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unraphael)](https://pypi.org/project/unraphael/)
[![PyPI](https://img.shields.io/pypi/v/unraphael.svg?style=flat)](https://pypi.org/project/unraphael/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11192044.svg)](https://doi.org/10.5281/zenodo.11192044)

![Unraphael banner](https://raw.githubusercontent.com/DecodingRaphael/unraphael/main/src/unraphael/data/logo.png#gh-light-mode-only)
![Unraphael banner](https://raw.githubusercontent.com/DecodingRaphael/unraphael/main/src/unraphael/data/logo-dark.png#gh-dark-mode-only)

# Unraphael

**Unraphael** is a digital workflow tool that uses computer vision to unravel the artistic practice of Raphael (Raffaello Sanzio, 1483-1520), while providing new digital approaches for the study of artistic practice in art history. Dozens of faithful reproductions survive of Raphael's paintings, attesting to the lucrative practice of serial production of paintings within the artist's workshop and to the lasting demand for the master's designs.

**Unraphael** provides a flexible and easy-to-use GUI to inspect and assess the structural similarity of figure-outlines in images. Photographs of paintings are used as input for the application.

While **Unraphael** was made for art historians and researchers in the humanities to study the artistic practices of and the process of making copies of paintings, the functionality of **Unraphael** extends well beyond the study of Raphael's paintings and can be used for a wide range of applications in the digital humanities and beyond.

Features:
- Image preprocessing
- Background removal
- Image alignment
- Image clustering based on structural similarity
- Ratio analysis painting - photo dimensions

To install:

```console
pip install unraphael
```

## Try unraphael in your browser!

You can also [try unraphael directly from your browser](https://unraphael.streamlit.app/).

<table>
  <tr>
    <th>Link</th>
    <th>Description</th>
    <th>Image</th>
  </tr>
  <tr>
    <td><a href =https://unraphael.streamlit.app/image_similarity>Image similarity</a></td>
    <td>Group your images using cluster analysis</td>
    <td><img src="https://raw.githubusercontent.com/DecodingRaphael/unraphael/main/docs/_static/dash_image_sim.png" alt="Image similarity" width="90%"/></td>
  </tr>
  <tr>
    <td><a href =https://unraphael.streamlit.app/image_similarity>Image preprocessing</a></td>
    <td>Preprocess your images, e.g. background removal, color adjustments, applying image filters, segmentation</td>
    <td><img src="https://raw.githubusercontent.com/DecodingRaphael/unraphael/main/docs/_static/dash_preprocess.png" alt="Image preprocessing" width="90%"/></td>
  </tr>
  <tr>
    <td><a href =https://unraphael.streamlit.app/image_similarity>Object detection</a></td>
    <td>Quickly and accurately identify and segment figures or objects within an image to analyse the isolated components</td>
    <td><img src="https://raw.githubusercontent.com/DecodingRaphael/unraphael/main/docs/_static/dash_detect.png" alt="Object detection" width="90%"/></td>
  </tr>
  <tr>
    <td><a href =https://unraphael.streamlit.app/image_similarity>Image comparison</a></td>
    <td>Compare your images based on their structural components</td>
    <td><img src="https://raw.githubusercontent.com/DecodingRaphael/unraphael/main/docs/_static/dash_compare.png" alt="Image comparison" width="90%"/></td>
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

## Self hosted deployment

To run on dashboard with [uv](https://docs.astral.sh/uv/) installed:

```shell
uvx -p 3.12 --from \
'unraphael[dash]@git+https://github.com/DecodingRaphael/unraphael.git@0.3' \
unraphael-dash
```

Healthcheck url at http://localhost:8501/_stcore/health

<details>
  <summary>Systemd service</summary>

To run unraphael as a service, you can create a systemd service file. This will allow you to start, stop, and restart unraphael using systemd.

1.  Create a service file for unraphael, for example `/etc/systemd/system/unraphael.service`:

```
[Unit]
Description=Unraphael dashboard
After=network.target

[Service]
User=youruser
WorkingDirectory=/home/youruser
ExecStart=/home/youruser/.local/bin/unraphael-dash
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Replace `youruser` with your actual username.  Also, make sure that the path to `unraphael-dash` is correct. You can find the correct path using `which unraphael-dash`.

2.  Enable the service:

```console
sudo systemctl enable unraphael.service
```

3.  Start the service:

```console
sudo systemctl start unraphael.service
```

4.  Check the status of the service:

```console
sudo systemctl status unraphael.service
```

5.  To stop the service:

```console
sudo systemctl stop unraphael.service
```

</details>
