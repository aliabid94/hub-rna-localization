"""
Utilities for models, e.g. training them and evaluating performance
"""
import numpy as np


def list_preds_to_array_preds(list_preds):
    """
    Given a list of preds, with each item in the list corresponding for predictions for a class return
    as a matrix of preds. This assumes that the preds are for a binary classification task at each class.
    """
    assert isinstance(list_preds, list), f"Unrecognized input type: {type(list_preds)}"
    pos_preds = []
    for i in range(len(list_preds)):
        pos_preds.append(np.atleast_2d(list_preds[i][:, 1]).T)
    retval = np.hstack(pos_preds)
    assert retval.shape == (list_preds[0].shape[0], len(list_preds))  # num_examples x num_classes
    return retval

