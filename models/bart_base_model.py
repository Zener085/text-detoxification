"""
Functions for using a bart-based-detox model.
It's recommended to use another dataset for tokenizer, pay attention to this before using that.
"""
__author__ = "Zener085"
__version__ = "1.0.0"
__license__ = "MIT"
__all__ = ["tokenizer", "bart_detox"]

from transformers import BartForConditionalGeneration, BartTokenizerFast
from torch import Tensor
from typing import Union

__MAX_INPUT_LENGTH = 1500
__MAX_OUTPUT_LENGTH = 1000
__MODEL = BartForConditionalGeneration.from_pretrained("SkolkovoInstitute/bart-base-detox").to("cuda")

tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")


def bart_detox(__text: Union[str, Tensor]) -> str:
    if isinstance(__text, str):
        __text = tokenizer.encode(__text, return_tensors="pt", max_length=__MAX_INPUT_LENGTH, truncation=True)
    __text = __text.to(__MODEL.device)
    _output = __MODEL.generate(__text, max_length=__MAX_OUTPUT_LENGTH, num_return_sequences=1, do_sample=True)[0]
    return tokenizer.decode(_output, skip_special_tokens=True)
