from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
import yaml

from unraphael.locations import data_directory

if TYPE_CHECKING:
    from io import IOBase


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


@st.cache_data
def _load_config_file(fn: Path | str | IOBase) -> dict:
    """Load config dict from file."""
    if isinstance(fn, (Path, str)):
        with open(fn) as f:
            config = yaml.safe_load(f)
    else:
        config = yaml.safe_load(fn)

    return config


def load_config(fn: Path | str | IOBase | None = None) -> dict:
    """Load config into session state and return copy."""
    if not fn:
        fn = data_directory / 'config.yaml'  # type: ignore

    config = _load_config_file(fn)

    if 'config' not in st.session_state:
        st.session_state.config = config

    return config
