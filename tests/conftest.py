"""Pytest shared-config for pyECM suite"""


def pytest_collection_modifyitems(config, items):
    """Run tests marked as 'long' at last

    """
    long_tests = [item for item in items if item.get_closest_marker("long")]
    other_tests = [item for item in items if not item.get_closest_marker("long")]
    items[:] = other_tests + long_tests