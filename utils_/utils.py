from functools import wraps
import time


def timing_wrapper(message):
    def f(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print(message, f': {int(time.time() - start)} seconds.')
            return res

        return wrapper

    return f
