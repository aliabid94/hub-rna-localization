"""
Various miscellaneous utility functions
"""
import os
import logging
import pickle
import sklearn


def load_sklearn_model(file_name:str, disable_multiprocessing:bool=False, strict:bool=True):
    """
    Load the sklearn model from the given filename
    """
    logging.info(f"Loading sklearn model from {file_name}")
    if strict:
        bname = os.path.basename(file_name)
        tokens = bname.split(".")
        version = '.'.join(tokens[1:-1])
        assert sklearn.__version__ == version, f"Got mismatched sklearn versions: Installed: {sklearn.__version__} Expected: {version}"
    with open(file_name, 'rb') as source:
        retval = pickle.load(source)
    if disable_multiprocessing:
        retval.n_jobs = 1
    return retval

