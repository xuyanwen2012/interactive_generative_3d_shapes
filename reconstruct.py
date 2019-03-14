#!/usr/bin/env python3
"""Reconstructs an obj file from its json surface-depth parameterization.

Usage:
  reconstruct.py json <file>
"""
from docopt import docopt
import numpy as np
import json
import os

def reconstruct_mesh (data):
    print(data.shape)
    enforce(type(data) == np.ndarray, "expected a numpy array, not %s", type(data))
    enforce(len(data.shape) == 1, "expected a 1d array, not shape %s"%(data.shape,))
    print(data)
    return "fubar"

def write_file (result, output_file):
    basedir, file = os.path.split(output_file)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    with open(output_file, 'w') as f:
        f.write(result)

def process_json_file (input_file, output_file):
    with open(input_file, 'r') as f:
        data = np.array(json.loads(f.read()))
        print(data.shape)
        result = reconstruct_mesh(data)
        write_file(result, output_file)

def enforce(condition, fmt, *args, exception=Exception):
    if not condition:
        raise exception(fmt % args)

if __name__ == '__main__':
    args = docopt(__doc__)

    # validate arguments
    class ArgumentParsingException(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    enforce_arg = lambda cond, fmt, *args: enforce(cond, fmt, *args, exception=ArgumentParsingException)

    try:
        if args['<file>']:
            input_file = args['<file>']
            if args['json']:
                enforce(input_file.endswith(".json"), "%s is not a json file", input_file)
            enforce(os.path.exists(input_file), "%s does not exist", input_file)

    except ArgumentParsingException as e:
        print("Invalid argument: %s" % e)
        sys.exit(-1)

    if args['json']:
        basedir, file = os.path.split(input_file)
        process_json_file(input_file, os.path.join('reconstructed', file.replace('.json', '.obj')))
