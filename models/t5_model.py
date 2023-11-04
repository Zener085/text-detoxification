"""
Text detoxification by using t5 model.
Firstly, I wanted to get t5-11b model, but then I found I need to give 45gb of memory for that (I have only 30).
Thus, I used some lighter models, but by default I passed the largest one.
If you don't want to crush your computer and have a little free memory, be careful to change the model name before using
this file.
"""
__author__ = "Zener085"
__version__ = "1.0.0"
__license__ = "MIT"
__all__ = ["tokenizer", "t5_detox"]

import transformers
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from torch import Tensor
from typing import Union

transformers.set_seed(42)

__MAX_INPUT_LENGTH = 1500
__MAX_OUTPUT_LENGTH = 1000
__MODEL = T5ForConditionalGeneration.from_pretrained("t5-base").to("cuda")

tokenizer = T5TokenizerFast.from_pretrained("t5-base")


def t5_detox(__text: Union[str, Tensor]) -> str:
    """
    Detoxifies text using t5 pretrained model.

    Args:
        __text: A text that must be detoxified.

    Returns:
        A new text, detoxified.
    """
    if isinstance(__text, str):
        __text = tokenizer.encode(__text, return_tensors="pt", max_length=__MAX_INPUT_LENGTH, truncation=True)
    __text = __text.to(__MODEL.device)
    _output = __MODEL.generate(__text, max_length=__MAX_OUTPUT_LENGTH, num_return_sequences=1, do_sample=True)[0]
    return tokenizer.decode(_output, skip_special_tokens=True)
