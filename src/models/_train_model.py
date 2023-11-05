"""
Functionality to train the model using the main dataset
"""
__author__ = "Zener085"
__version__ = "1.0.0"
__license__ = "MIT"
__all__ = ["train_model"]

from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


def _preprocess(__input_text, __output_text, __tokenizer):
    """
    Preprocess single element from the dataset.

    Args:
        __input_text: Text with toxicity.
        __output_text: Nontoxic text.
        __tokenizer: Tokenizer for the trained model.

    Returns:
        Input data for the model.
    """
    _prefix = "Detoxify: "
    _max_input_length = 1500
    _max_output_length = 1000

    __input_text = _prefix + __input_text
    __input_tensor = __tokenizer(__input_text, return_tensors="pt", max_length=_max_input_length, truncation=True)

    __output_tensor = __tokenizer(__output_text, return_tensors="pt", max_length=_max_output_length, truncation=True)

    __input_tensor["labels"] = __output_tensor["input_ids"][0]
    __input_tensor["input_ids"] = __input_tensor["input_ids"][0]
    __input_tensor["attention_mask"] = __input_tensor["attention_mask"][0]
    return __input_tensor


def train_model(__model, __tokenizer, __df):
    """
    Trains given model using provided data and tokenizer for the model.

    Args:
        __model: A model that must be trained.
        __tokenizer: A tokenizer for the model.
        __df: Dataset of data.
              By default, it uses the initial dataset.
              If you want to use another - recreate preprocess stuff.

    Returns:
        Trained model.
    """
    _dataset = []
    for i in range(len(__df)):
        _dataset.append(_preprocess(__df["reference"][i], __df["translation"][i], __tokenizer))

    _split = 0.2
    _text_train, _text_test = train_test_split(_dataset, test_size=_split, shuffle=True)
    _batch_size = 32
    _args = Seq2SeqTrainingArguments(
        "Model fine-tuning",
        evaluation_strategy="epoch",
        learning_rate=0.001,
        per_device_train_batch_size=_batch_size,
        per_device_eval_batch_size=_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        report_to="tensorboard"
    )

    _data_collator = DataCollatorForSeq2Seq(__tokenizer, model=__model)

    _trainer = Seq2SeqTrainer(
        __model,
        _args,
        train_dataset=_text_train,
        eval_dataset=_text_test,
        data_collator=_data_collator,
        tokenizer=__tokenizer
    )
    _trainer.train()
    _trainer.save_model("../../models/best")
