#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import wandb
import torch
import constants


import transformers.src.transformers
from transformers.src.transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    GPT2Config,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    BertTokenizer,
    HfArgumentParser,
    Trainer_Time,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizer,
    DataCollatorForTimeControl,
    WikisectionDataset,
    # EekeDataset,
    TaskmasterDataset,
    GPT2TimeLMHeadModel,
)
from my_datasets.lm_dataset import EekeDataset
from transformers.src.transformers.trainer_utils import get_last_checkpoint
from transformers.src.transformers.utils import check_min_version
from transformers.src.transformers.utils.versions import require_version

import torch.nn as nn
from model import language

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    dryrun: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_contrastive_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_section_ids: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    embedding_type: str = field(
        default="entireSection",
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    encoder_filepath: str = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    latent_dim: int = field(
        default=32,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    project: Optional[str] = field(
        default="lm", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    use_bos: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_section_null: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    label:str = field(
        default="",
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def get_data_paths(data_args: DataTrainingArguments):
    """Datasets:

    """
    assert data_args.dataset_name in [
        'wikisection', 'roc_stories', 'wikihow', 'recipe', 'erke', 'tickettalk']
    # Default datapaths
    train_path = os.path.join(constants.PATH2WIKISECTION, "wikisection_withSections.train.txt")
    val_path = os.path.join(constants.PATH2WIKISECTION, "wikisection_withSections.val.txt")
    test_path = os.path.join(constants.PATH2WIKISECTION, "wikisection_withSections.test.txt")
    if "erke" in data_args.dataset_name:
        train_path = os.path.join(constants.PATH2Erke, "train-.json")
        val_path = os.path.join(constants.PATH2Erke, "valid-.json")
        test_path = os.path.join(constants.PATH2Erke, "valid-.json")
    return train_path, val_path, test_path

def get_dataset(
    cl_model,
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    file_path: str,
    special_words: list,
    data_names,
):
    if "wikisection" in args.dataset_name:
        dataset = WikisectionDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            use_section_null=args.use_section_null,
            special_words=special_words,
            block_size=args.block_size,
            cl_model=cl_model
        )
    elif 'tickettalk' in args.dataset_name:
        dataset = TaskmasterDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            use_section_null=args.use_section_null,
            special_words=special_words,
            block_size=args.block_size,
            cl_model=cl_model,
            data_names=data_names
        )
    elif 'erke' in args.dataset_name:
        dataset = EekeDataset(
            tokenizer=tokenizer,
            use_section_null=args.use_section_null,
            special_words=special_words,
            block_size=args.block_size,
            cl_model=cl_model,
            data_names=data_names
        )
    return dataset

def get_data_collator(
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer):
    data_collator = DataCollatorForTimeControl(
        tokenizer=tokenizer)
    return data_collator

HIDDEN_DIM = 128

def get_checkpoint(dataset_name, latent_dim, base_model="gpt2",
                   use_section_ids=False, token_size=None,
                   filepath=None):
    '''
    加载布朗模型
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = language.GPT2OUEncoder(
        hidden_dim=HIDDEN_DIM,
        latent_dim=latent_dim,
        finetune_gpt2=False)
    if use_section_ids:
        model.model.resize_token_embeddings(token_size)

    transformers.__spec__ = 'gpt2'  # Avoid bug
    state_dict = torch.load(filepath)
    new_dict = {}
    for k, v in state_dict['state_dict'].items():
        if any([i in k for i in ['model.model.g_ar', 'model.model.W_k']]):
            new_dict[k[6:]] = v
        elif any([i in k for i in ['model.g_ar', 'model.W_k', 'time_model']]):
            continue
        elif "model." in k:
            new_dict[k[6:]] = v
        else:
            new_dict[k] = v

    if any(['g_ar' in k for k in new_dict.keys()]):
        model.g_ar = nn.GRU(input_size=latent_dim,
                            hidden_size=2400,  # default number in infoNCE for langauge
                            num_layers=3,
                            batch_first=True
                            )
        model.W_k = nn.Linear(2400, latent_dim)
    elif any(['time_model' in k for k in state_dict['state_dict'].keys()]):
        model.fc_mu = nn.Linear(latent_dim, latent_dim)
        model.fc_var = nn.Linear(latent_dim, latent_dim)

    model.load_state_dict(new_dict)
    for p in model.parameters():
        p.requires_grad = False

    model.to(device)
    model = model.eval()
    return model


def get_special_tokens(dataset_name, tokenizer, add_tokens=True):
    SECTION_IDS = []
    if 'erke' in dataset_name:
        SECTION_IDS = [
            '[ user ]',
            '[ assistant ]',
            '<|endoftext|>'
        ]

    if 'tickettalk' in dataset_name:
        SECTION_IDS = [
            '[ USER ]',
            '[ ASSISTANT ]',
        ]

    SECTION_IDS += [' . ']
    if add_tokens:
        # NOTE loading previous tokenizer sometimes already includes the new tokens
        eos = tokenizer(' . ')['input_ids']
        print("Old tokenizer size: ", len(tokenizer))
        if len(eos) == 1 and eos[0] == 21128 + len(SECTION_IDS):
            print('========================================================================')
            print("Not adding because it's already contained")
            pass # don't add cause it's already contained
        else:
            print("Adding tokens, ", SECTION_IDS)
            tokenizer.add_tokens(SECTION_IDS)
        print("New tokenizer size: ", len(tokenizer))
    SPECIAL_TOKENS = [_[1] for _ in tokenizer(SECTION_IDS)['input_ids']]
    return SECTION_IDS, SPECIAL_TOKENS, tokenizer

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if model_args.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'

    os.environ['WANDB_CONSOLE']='wrap'

    if data_args.project is not None:
        os.environ['WANDB_PROJECT'] = data_args.project

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)sfilter_ -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.src.transformers.utils.logging.set_verbosity_info()
        transformers.src.transformers.utils.logging.enable_default_handler()
        transformers.src.transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)


    model_args.cache_dir = constants.PATH2HUGGINGFACE
    gpt2_path = constants.PATH2GPT

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = GPT2Config.from_pretrained(gpt2_path, **config_kwargs)

        print('=============================there there there========================')
        print(gpt2_path)
        print(config)

    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    config.use_contrastive_embeddings = model_args.use_contrastive_embeddings
    config.embedding_type = model_args.embedding_type
    config.use_section_ids = model_args.use_section_ids
    config.use_section_null = data_args.use_section_null
    config.dataset_name = data_args.dataset_name
    config.cl_latent_dim = model_args.latent_dim


    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    # tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, **tokenizer_kwargs)
    tokenizer = BertTokenizer.from_pretrained(gpt2_path, **tokenizer_kwargs)

    SECTION_IDS, SPECIAL_TOKENS, tokenizer = get_special_tokens(
        dataset_name=data_args.dataset_name, tokenizer=tokenizer)
    # -1 because of the added " . "
    config.max_num_sections = len(SECTION_IDS) - 1

    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.bos_token = '<|endoftext|>'
    tokenizer.pad_token = tokenizer.eos_token

    config.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>') # 修改id 50256->21130  hugging默认是50256，即英文模型大小
    config.bos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>') # 修改id 50256->21130

    print('=============================here here here========================')
    print(config)


    if model_args.model_name_or_path:
        model = GPT2TimeLMHeadModel.from_pretrained(
            gpt2_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))
    model.transformer.special_tokens = SPECIAL_TOKENS
    print("Resized model to {}".format(len(tokenizer)))
    print("Added special tokens, ", SPECIAL_TOKENS)
    print("Added special ids, ", SECTION_IDS)

    # Getting checkpoint dict:
    cpu_device = torch.device('cpu')
    base_model = 'gpt2'
    CL_MODEL = get_checkpoint(
        dataset_name=data_args.dataset_name,
        latent_dim=model_args.latent_dim,
        use_section_ids=True,
        token_size= len(tokenizer),
        base_model=base_model,
        filepath=model_args.encoder_filepath
    )# .to(cpu_device)

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    data_args.block_size = block_size
    ### Data
    train_path, val_path, eval_path = get_data_paths(data_args)

    train_dataset = get_dataset(
        args=data_args, tokenizer=tokenizer,
        file_path=train_path,
        special_words=SECTION_IDS,
        cl_model=CL_MODEL,
        data_names='train'
    )
    eval_dataset = get_dataset(
        args=data_args, tokenizer=tokenizer,
        file_path=eval_path,
        special_words=SECTION_IDS,
        cl_model=CL_MODEL,
        data_names = 'eval'
    )

    data_collator = get_data_collator(
        args=data_args,
        tokenizer=tokenizer)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Initialize our Trainer
    trainer = Trainer_Time(
        model=model,
        special_tokens=SPECIAL_TOKENS,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        # data_collator=default_data_collator,
        data_collator=data_collator
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            perplexity = math.exp(metrics["train_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
            wandb.log({"ppl": perplexity})
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
