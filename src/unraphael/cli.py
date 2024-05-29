from __future__ import annotations

import logging
import sys

import toml

from .locations import dash_directory

logger = logging.getLogger(__name__)


def dash_entry(**kwargs):
    """Start streamlit dashboard."""
    from streamlit.web import cli as stcli

    argv = sys.argv[1:]

    dashboard_path = dash_directory / 'home.py'

    config_file = dash_directory / '.streamlit' / 'config.toml'
    config = toml.load(config_file)

    opts = []
    for group, options in config.items():
        for option, value in options.items():
            opts.extend((f'--{group}.{option}', str(value)))

    sys.argv = ['streamlit', 'run', str(dashboard_path), *opts, *argv]

    print(sys.argv)

    logger.debug('Streamlit arguments %s', sys.argv)

    sys.exit(stcli.main())
