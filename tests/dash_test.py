from __future__ import annotations

import sys

from streamlit.testing.v1 import AppTest

from unraphael.locations import dash_directory

sys.path.append(str(dash_directory))


def test_home():
    at = AppTest.from_file(str(dash_directory / 'home.py'))
    at.run(timeout=5)
    assert not at.exception


def test_preprocess_load():
    at = AppTest.from_file(str(dash_directory / 'pages' / '1_preprocess.py'))
    at.run(timeout=5)
    assert not at.exception


def test_image_sim_load():
    at = AppTest.from_file(str(dash_directory / 'pages' / '2_image_similarity.py'))
    at.run(timeout=5)
    assert not at.exception


def test_detect_objects_load():
    at = AppTest.from_file(str(dash_directory / 'pages' / '3_detect_objects.py'))
    at.run(timeout=5)
    assert not at.exception


def test_compare_load():
    at = AppTest.from_file(str(dash_directory / 'pages' / '4_compare.py'))
    at.run(timeout=5)
    assert not at.exception


def test_preprocess_workflow():
    at = AppTest.from_file(str(dash_directory / 'pages' / '1_preprocess.py'))
    at.run(timeout=5)
    assert not at.exception

    assert 'load_example' in at.session_state
    at.session_state['load_example'] = True
    at.run(timeout=10)

    assert not at.exception


def test_image_similarity_workflow():
    at = AppTest.from_file(str(dash_directory / 'pages' / '2_image_similarity.py'))
    at.run(timeout=5)
    assert not at.exception

    assert 'continue_ransac' not in at.session_state
    assert 'load_example' in at.session_state

    at.session_state['load_example'] = True
    at.run(timeout=5)

    assert not at.exception

    assert 'continue_ransac' in at.session_state
    assert at.session_state['method'] == 'sift'

    at.session_state['continue_ransac'] = True
    at.run(timeout=10)

    assert not at.exception


def test_detect_objects_workflow():
    at = AppTest.from_file(str(dash_directory / 'pages' / '2_image_similarity.py'))
    at.run(timeout=5)
    assert not at.exception

    assert 'load_example' in at.session_state

    at.session_state['load_example'] = True
    at.run(timeout=5)

    assert not at.exception

    at.session_state['select task'] = 'Detection'
    at.run(timeout=10)

    assert not at.exception


def test_compare_workflow():
    at = AppTest.from_file(str(dash_directory / 'pages' / '4_compare.py'))
    at.session_state['width'] = 100
    at.run(timeout=5)
    assert not at.exception

    at.session_state['load_example'] = True
    at.run(timeout=5)

    assert not at.exception

    at.session_state['alignment procedure'] = 'Feature based alignment'
    at.run(timeout=10)

    assert not at.exception
