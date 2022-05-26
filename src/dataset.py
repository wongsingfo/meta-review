from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer
from typing import cast
import logging
import multiprocessing

logger = logging.getLogger(__name__)

def prepare_dataset(data_args, model_args) -> DatasetDict:
    # if use_dataloader is False:
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        extension = None
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        assert extension is not None
        datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir
        )

    datasets = cast(DatasetDict, datasets)
    return datasets


def get_preprocess_function(
        tokenizer,
        max_source_length,
        max_target_length,
        padding,
        prefix="",
        text_column="text",
        summary_column="summary",
        ignore_pad_token_for_loss=True,
):
    def preprocess(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs] # type: list[str]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True) # BatchEncoding
        # model_inputs[i]: Encoding

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                # tokenizer.pad_token_id = 1
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    return preprocess

def preprocess_dataset(dataset, processosr, max_samples, remove_columns) -> DatasetDict:
    if max_samples:
        dataset = dataset.select(range(max_samples))
    dataset = dataset.map(
        processosr,
        batched=True,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        # num_proc=multiprocessing.cpu_count(),
    )
    return dataset

def get_dataset(data_args, model_args, training_args, tokenizer) -> tuple[DatasetDict, DatasetDict, DatasetDict]:
    datasets = prepare_dataset(data_args, model_args)

    if training_args.do_train:
        column_names = datasets["train"].column_names # pyright: ignore
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names # pyright: ignore
    elif training_args.do_predict:
        column_names = datasets["test"].column_names # pyright: ignore
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        exit(1)
    # Get the column names for input/target.

    text_column, summary_column = column_names
    # Temporarily set max_target_length for training.
    padding = "max_length" if data_args.pad_to_max_length else False

    processosr = get_preprocess_function(
        tokenizer,
        data_args.max_source_length,
        data_args.max_target_length,
        padding,
        prefix=data_args.source_prefix or "",
        text_column=text_column,
        summary_column=summary_column,
        ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss
    )

    train_dataset = training_args.do_train and preprocess_dataset(
        datasets['train'], processosr, data_args.max_train_samples, column_names)
    eval_dataset = training_args.do_eval and preprocess_dataset(
        datasets['validation'], processosr, data_args.max_eval_samples, column_names)
    test_dataset = training_args.do_predict and preprocess_dataset(
        datasets['test'], processosr, data_args.max_predict_samples, column_names)

    return train_dataset, eval_dataset, test_dataset
