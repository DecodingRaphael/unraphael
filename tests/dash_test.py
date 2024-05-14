from streamlit.testing.v1 import AppTest
from pathlib import Path
import sys
import pytest

dashdir = Path(__file__).parents[1] / 'dash'
sys.path.append(str(dashdir))


def test_home():
    at = AppTest.from_file(str(dashdir / 'home.py'))
    at.run()
    assert not at.exception


@pytest.mark.xfail(
    reason='Fails with Thread "MainThread": missing ScriptRunContext'
    'on the CI because of rembg dependency. Only the first time the page is'
    'loaded does the test fail.'
)
def test_preprocess_load():
    at = AppTest.from_file(str(dashdir / 'pages' / '1_preprocess.py'))
    at.run()
    assert not at.exception


def test_image_sim_load():
    at = AppTest.from_file(str(dashdir / 'pages' / '2_image_similarity.py'))
    at.run()
    assert not at.exception


def test_preprocess_workflow():
    at = AppTest.from_file(str(dashdir / 'pages' / '1_preprocess.py'))
    at.run()
    assert not at.exception

    assert 'load_example' in at.session_state
    at.session_state['load_example'] = True
    at.run(timeout=10)

    assert not at.exception


def test_image_similarity_workflow():
    at = AppTest.from_file(str(dashdir / 'pages' / '2_image_similarity.py'))
    at.run()
    assert not at.exception

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
