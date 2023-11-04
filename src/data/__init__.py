"""
Functions for creating and preprocessing data for the training and predicting.
"""
__author__ = "Zener085"
__version__ = "1.0.0"
__license__ = "MIT"
__all__ = ["load_main_dataset", "preprocess_dataset"]

import pandas as pd
from os.path import exists
from torch import Tensor
from typing import List, Tuple


def load_main_dataset() -> pd.DataFrame:
    """
    Loads filtered.tsv file that used for the training the model.
    Deletes and changes a bit some information inside this dataset.

    Returns:
        Prepared dataset, with no preprocessing.
    """
    if exists("../../data/raw/filtered.tsv"):
        _df = pd.read_csv("../../data/raw/filtered.tsv", sep="\t")
    else:
        _df = pd.read_csv("../data/raw/filtered.tsv", sep="\t")

    _df.drop("Unnamed: 0", inplace=True, axis=1)

    _swap = _df["ref_tox"] < _df["trn_tox"]
    _df.loc[_swap, ["reference", "translation", "ref_tox", "trn_tox"]] = (
        _df.loc[_swap, ["translation", "reference", "trn_tox", "ref_tox"]].values
    )

    return _df


def preprocess_dataset(__df: pd.DataFrame, __tokenizer, __device: str = "cpu") -> Tuple[List[Tensor], List[List[int]]]:
    """
    Preprocesses the dataframe for the model training. Works only with base dataset.

    Parameters:
        __df: Dataframe to be preprocessed.
        __tokenizer: Tokenizer for the text.
        __device: To which device the data must be converted.

    Returns:
        Preprocessed dataframe.
    """
    _prefix = "Detoxify: "
    _max_input_length = 1500
    _max_output_length = 1000

    _inputs = [_prefix + _text for _text in __df["reference"]]
    _inputs = [__tokenizer.encode(_input, return_tensors="pt", max_length=_max_input_length, truncation=True)
               for _input in _inputs]

    _outputs = [[_text] for _text in __df["translation"]]

    return _inputs, _outputs
