import logging

import pytorch_lightning as pl
from torch import nn

from transformers import AutoConfig, AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)


class PretrainedTransformer(pl.LightningModule):
    def __init__(self, args, num_labels=None, mode="base", **config_kwargs):
        "Initialize a model."

        super().__init__()
        self.args = args
        cache_dir=self.args.cache_dir if self.args.cache_dir else None
        self.config = AutoConfig.from_pretrained(
            self.args.config_name if self.args.config_name else self.args.model_name_or_path,
            **({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=cache_dir,
            **config_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_name if self.args.tokenizer_name else self.args.model_name_or_path,
            cache_dir=cache_dir,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            self.args.model_name_or_path,
            from_tf=bool(".ckpt" in self.args.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam for the non-transformer parameters.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_ratio", default=0, type=float, help="The fraction of steps to warm up.")
        parser.add_argument(
            "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
        )

        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
