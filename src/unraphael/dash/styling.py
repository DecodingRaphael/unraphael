from __future__ import annotations

import base64

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


@st.cache_data
def get_base64_of_bin_file(png_file):
    with open(png_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(
    png_file,
    background_position='50% 10%',
    margin_top='10%',
    image_width='60%',
    image_height='',
    nav=True,  # page has navigation
):
    binary_string = get_base64_of_bin_file(png_file)

    loc = 'stSidebarHeader' if nav else 'stSidebarUserContent'

    return f"""
            <style>
                [data-testid="{loc}"] {{
                    background-image: url("data:image/png;base64,{binary_string}");
                    background-repeat: no-repeat;
                    background-position: {background_position};
                    margin-top: {margin_top};
                    background-size: {image_width} {image_height};
                }}
            </style>
            """


def add_sidebar_logo():
    """Based on: https://stackoverflow.com/a/73278825."""
    png_file = data_directory / 'logo-dark.png'
    logo_markup = build_markup_for_logo(png_file, nav=True)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )
