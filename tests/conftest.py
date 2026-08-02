"""Pytest shared-config for pyECM suite"""


def pytest_collection_modifyitems(config, items):
    """Run tests marked as 'long' at last

    """
    long_tests = [item for item in items if item.get_closest_marker("long")]
    network_tests = [item for item in items if item.get_closest_marker("network") and item not in long_tests]
    other_tests = [item for item in items if item not in long_tests and item not in network_tests]
    items[:] = other_tests + network_tests + long_tests
