import numpy as np
import pytorch_lightning as pl
import torch

from transformers import get_linear_schedule_with_warmup
from transformers import AdamW


class BaseLightningModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self, shuffle=True, use_lr_scheduler=True):
        train_batch_size = self.args.train_batch_size
        dataloader = self.load_dataset("train", train_batch_size, shuffle=shuffle)

        if use_lr_scheduler:
            t_total = (
                (len(dataloader.dataset) // (train_batch_size * max(1, self.args.gpus)))
                // self.args.gradient_accumulation_steps
                * float(self.args.num_train_epochs)
            )
            scheduler = get_linear_schedule_with_warmup(
                self.opt, num_warmup_steps=self.args.warmup_ratio * t_total, num_training_steps=t_total
            )
            self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self, shuffle=False):
        dataloaders = [self.load_dataset("dev", self.args.eval_batch_size, shuffle=shuffle)]
        if self.has_secondary_split:
            dataloaders.append(self.load_dataset("dev2", self.args.eval_batch_size, shuffle=shuffle))
        return dataloaders

    def test_dataloader(self, shuffle=False):
        dataloaders = [self.load_dataset("test", self.args.eval_batch_size, shuffle=shuffle)]
        if self.has_secondary_split:
            dataloaders.append(self.load_dataset("test2", self.args.eval_batch_size, shuffle=shuffle))
        return dataloaders

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        assert not on_tpu, "No TPU support yet"
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        avg_loss = getattr(self.trainer, "avg_loss", 0.0)
        return {"loss": "{:.3f}".format(avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    def _eval_epoch_end(self, outputs, has_labels=False):
        if has_labels:
            val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()

        logits = np.concatenate([x["output_dict"]["logits"] for x in outputs], axis=0)
        if self.output_mode == "classification":
            preds = np.argmax(logits, axis=1)
        elif self.output_mode == "regression":
            preds = np.squeeze(logits)

        if has_labels:
            targets = np.concatenate([x["target"] for x in outputs], axis=0)
            return {**{"val_loss": val_loss_mean}, **self.compute_metrics(preds, targets)}
        else:
            return preds, logits

    def validation_epoch_end(self, outputs: list) -> dict:
        if self.has_secondary_split:
            # Each dataloader has its own individual metrics suffixed with its index,
            # and we also aggregate (average) metrics across dataloaders
            logs = {}
            sum_logs = {}
            for i, one_split_outputs in enumerate(outputs):
                one_split_logs = self._eval_epoch_end(one_split_outputs, has_labels=True)
                for k, v in one_split_logs.items():
                    logs[k + str(i + 1)] = v
                    if k not in {'preds', 'targets'}:
                        if k not in sum_logs:
                            sum_logs[k] = 0
                        sum_logs[k] += v
            logs.update({k: v / len(outputs) for k, v in sum_logs.items()})
        else:
            logs = self._eval_epoch_end(outputs, has_labels=True)
        self.validation_results = dict(logs)
        logs.update({"log": {**logs}, "progress_bar": {**logs}})
        return logs

    def test_epoch_end(self, outputs):
        if self.has_secondary_split:
            predictions = [self._eval_epoch_end(one_split_outputs) for one_split_outputs in outputs]
        else:
            predictions = self._eval_epoch_end(outputs)
        if getattr(self, 'include_logits_in_prediction', False):
            self.predictions = predictions
        else:
            self.predictions = [prediction[0] for prediction in predictions] if self.has_secondary_split else predictions[0]
        return {}
