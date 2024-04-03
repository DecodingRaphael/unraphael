from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
import yaml

if TYPE_CHECKING:
    import numpy as np


def show_images(images: dict[str, np.ndarray], *, n_cols: int = 4):
    cols = st.columns(n_cols)

    for i, (name, im) in enumerate(images.items()):
        col = cols[i % n_cols]
        col.image(im, use_column_width=True, caption=name)


def to_session_state(key: str, section: str | None = None):
    section = section if section else key

    try:
        st.session_state.config[section] = yaml.safe_load(st.session_state[key])
    except AttributeError:
        st.session_state.config[section] = st.session_state[key]
    except KeyError:
        print(f'Could not commit {key}->{section} in config')


def dump_session_state():
    return dump_config(st.session_state.config)


def dump_config(cfg: dict) -> str:
    def cfg_str_representer(dumper, in_str):
        if '\n' in in_str:  # use '|' style for multiline strings
            return dumper.represent_scalar('tag:yaml.org,2002:str', in_str, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', in_str)

    yaml.representer.SafeRepresenter.add_representer(str, cfg_str_representer)
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def _update_session_state(config: dict):
    st.session_state.config = config
    st.session_state.width = config['width']
    st.session_state.method = config['method']


@st.cache_data
def _load_config(fn: Path | str):
    fn = Path(fn)

    if not fn.exists():
        st.error(f'Cannot find {fn}.')

    with open(fn) as f:
        config = yaml.safe_load(f)

    _update_session_state(config)

    return config
