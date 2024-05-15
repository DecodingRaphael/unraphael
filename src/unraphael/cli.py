from importlib.resources import files
import toml
import logging
import sys


logger = logging.getLogger(__name__)


def dash_entry(**kwargs):
    """Start streamlit dashboard."""
    from streamlit.web import cli as stcli

    argv = sys.argv[1:]

    dashboard_path = files('unraphael.dash') / 'home.py'

    config_file = files('unraphael.dash') / '.streamlit' / 'config.toml'
    config = toml.load(config_file)

    opts = []
    for group, options in config.items():
        for option, value in options.items():
            opts.extend((f'--{group}.{option}', str(value)))

    sys.argv = ['streamlit', 'run', str(dashboard_path), *opts, *argv]

    print(sys.argv)

    logger.debug('Streamlit arguments %s', sys.argv)

    sys.exit(stcli.main())
