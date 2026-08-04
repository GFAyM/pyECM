"""Decorators shared across pyECM modules."""

import functools
import time


def debug_timed(label):
    """Print the wrapped function's wall-clock time.

    Only prints when the wrapped function is called with debug > 0.

    :param label: name shown in the printed message (e.g. "NR", "X2C")
    :type label: str
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            debug = kwargs.get("debug", 0)
            start = time.time()
            result = func(*args, **kwargs)
            if debug > 0:
                print(f"{label} time (min):", (time.time() - start) / 60)
            return result

        return wrapper

    return decorator
