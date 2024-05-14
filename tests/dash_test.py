from streamlit.testing.v1 import AppTest
from pathlib import Path
import pytest
import sys

dashdir = Path(__file__).parents[1] / 'dash'
sys.path.append(str(dashdir))

pages = list((dashdir / 'pages').glob('*.py'))


def test_home():
    at = AppTest.from_file(str(dashdir / 'home.py'))
    at.run(timeout=10)
    assert not at.exception


@pytest.mark.parametrize('page', pages)
def test_pages(page):
    at = AppTest.from_file(str(page))
    at.run(timeout=10)
    assert not at.exception


def test_preprocess():
    at = AppTest.from_file(str(dashdir / 'pages' / '1_preprocess.py')).run()

    assert 'load_example' in at.session_state
    at.session_state['load_example'] = True
    at.run(timeout=10)

    assert not at.exception


def test_image_similarity():
    at = AppTest.from_file(str(dashdir / 'pages' / '2_image_similarity.py')).run()

    assert 'continue_ransac' not in at.session_state
    assert 'load_example' in at.session_state

    at.session_state['load_example'] = True
    at.run()

    assert not at.exception

    assert 'continue_ransac' in at.session_state
    assert at.session_state['method'] == 'sift'

    at.session_state['continue_ransac'] = True
    at.run(timeout=10)

    assert not at.exception
