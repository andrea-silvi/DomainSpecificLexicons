import argparse
import json
import os
from functools import wraps
import time

def upload_args_from_json(file_path=os.path.join("parameters", "fixed_params.json")):
    parser = argparse.ArgumentParser(description=f'Arguments from json')
    args = parser.parse_args()
    json_params = json.loads(open(file_path).read())
    for option, option_value in json_params.items():
        if option_value == 'None':
            option_value = None
        setattr(args, option, option_value)
    setattr(args, "runNumber", 0)
    return args

def timing_wrapper(message):
    def f(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print(message, ': {int(time.time() - start)} seconds.')
            return res
        return wrapper
    return f
