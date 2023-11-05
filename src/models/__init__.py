"""
Scripts to train models and then use trained models to make predictions.
"""
__author__ = "Zener085"
__version__ = "1.0.0"
__license__ = "MIT"
__all__ = ["train", "detox"]

from ._detox_model import detox_model
from ._train_model import train_model

train, detox = train_model, detox_model
