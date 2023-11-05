"""
Functionality for predicting using some model.
"""
__author__ = "Zener085"
__version__ = "1.0.0"
__license__ = "MIT"
__all__ = ["detox_model"]

from transformers import AutoModelForSeq2SeqLM


def _detox_single(__model, __tokenizer, __text: str):
    """
    Generates nontoxic text from an input text.

    Args:
        __model: A model that is used for generation.
        __tokenizer: A tokenizer for the model.
        __text: Input text.
                Can be only `str` type.

    Returns:
        Nontoxic text.
    """
    _input_ids = __tokenizer(__text, return_tensors="pt").input_ids
    _outputs = __model.generate(input_ids=_input_ids)
    return __tokenizer.decode(_outputs[0], skip_special_tokens=True, temperature=0)


def detox_model(__tokenizer, __text):
    """
    Generates nontoxic text.

    Args:
        __tokenizer: Tokenizer for the model.
        __text: Input text for the model.
                It can be an iterable object.

    Returns:
        Nontoxic text.
        It can be either list of single output.
    """
    if isinstance(__text, str):
        __text = [__text]

    _model = AutoModelForSeq2SeqLM.from_pretrained("../../models/best")

    _nontoxic = []

    for __single_text in __text:
        _nontoxic.append(_detox_single(_model, __tokenizer, __single_text))

    if len(_nontoxic) == 1:
        return _nontoxic[0]

    return _nontoxic
