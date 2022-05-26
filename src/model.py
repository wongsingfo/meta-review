from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from typing import cast


def get_pretrained_model(data_args, model_args, token_embedding_size) -> PreTrainedModel:
    # Load pretrained model and tokenizer
    config: PretrainedConfig = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )

    # MRED: allow source length > 1024
    # We have to modify the transformers source code to allow length that larger than 1024
    if data_args.max_source_length > 1024:
        print("setting max position embedding:", data_args.max_source_length)
        config.max_position_embeddings = data_args.max_source_length  # pyright: ignore

    # MRED: configuring Bart to produce target between length of 20 to 400
    config.max_length = data_args.max_target_length
    config.min_length = model_args.gen_target_min

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = cast(PreTrainedModel, model)
    model.resize_token_embeddings(token_embedding_size)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    return model
