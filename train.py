import argparse
from importlib import reload
import logging
import os
import json
from pathlib import Path
import random

import numpy as np
import pytorch_lightning as pl
import torch

reload(logging)

from models import RGCNSemanticEncoder


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def __init__(self):
        self.best_epoch = None
        self.best_dev_metric = None
        self.best_dev_metrics = None

    def on_validation_end(self, trainer, pl_module):
        if pl_module.trainer.proc_rank <= 0:
            logger.info("")
            logger.info("***** Validation results *****")

            assert pl_module.metric_watch_mode in {'max', 'min'}

            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}".format(key, str(metrics[key])))

                if key == pl_module.metric_to_watch:
                    if (
                        self.best_dev_metric is None
                        or (pl_module.metric_watch_mode == 'max' and metrics[key] > self.best_dev_metric)
                        or (pl_module.metric_watch_mode == 'min' and metrics[key] < self.best_dev_metric)
                    ):
                        self.best_epoch = trainer.current_epoch
                        self.best_dev_metric = metrics[key]
                        self.best_dev_metrics = {
                            k: v for k, v in metrics.items() if k not in {"log", "progress_bar", "loss", "val_loss", "rate", "epoch"}
                        }

            logger.info(f"best_epoch = {self.best_epoch}")
            for key, value in sorted(self.best_dev_metrics.items()):
                logger.info(f"best_{key} = {value}")


class ModelCheckpointCallback(pl.callbacks.ModelCheckpoint):
    def _save_model(self, filepath):
        try:
            return super()._save_model(filepath)
        except (OSError, RuntimeError) as e:  # If we run out of disk space, we can carry on
            logging.warning(repr(e))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus > 0:
        torch.cuda.manual_seed_all(args.seed)


def add_generic_args(parser, root_dir):
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--n_tpu_cores", type=int, default=0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")


def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = RGCNSemanticEncoder.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        content = os.listdir(args.output_dir)
        # For DDP, when subprocesses are launched, there'll be a log.txt inside the folder already
        if len(content) > 0 and content != ['log.txt'] and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.mkdir(args.output_dir)

    json.dump(vars(args), open(os.path.join(args.output_dir, 'args.json'), 'w'))

    if args.gpus is None:
        args.gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if 'CUDA_VISIBLE_DEVICES' in os.environ else 0

    set_seed(args)

    # Set by pytorch-lightning
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))

    logging.basicConfig(
        level=logging.INFO if local_rank <= 0 else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    model = RGCNSemanticEncoder(args)

    checkpoint_callback = ModelCheckpointCallback(
        filepath=os.path.join(args.output_dir, f'{{epoch}}_{{{model.metric_to_watch}:.4f}}'),
        monitor=model.metric_to_watch,
        mode=model.metric_watch_mode,
        save_top_k=1,
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.gpus,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        gradient_clip_val=args.max_grad_norm,
        default_root_dir=args.output_dir,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
        replace_sampler_ddp=False,
    )

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_tpu_cores > 0:
        global xm
        import torch_xla.core.xla_model as xm

        train_params["num_tpu_cores"] = args.n_tpu_cores
        train_params["gpus"] = 0

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(**train_params)

    if args.do_train:
        trainer.fit(model)
        if local_rank <= 0:
            os.symlink(checkpoint_callback.best_model_path.split('/')[-1], Path(checkpoint_callback.best_model_path).parent / 'best.ckpt')

if __name__ == "__main__":
    main()
