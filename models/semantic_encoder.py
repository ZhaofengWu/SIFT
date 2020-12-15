import logging
import os
import pickle
from types import GeneratorType

from allennlp.nn.util import masked_mean
from dgl import DGLGraph
from pytorch_lightning.utilities import move_data_to_device
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.activations import ACT2FN

from data_readers import processors, output_modes, tasks_num_labels, secondary_processors
from data_readers.base_reader import convert_examples_to_features, get_graphs
from data_readers.semantic_dataset import SemanticDataset
from metrics import compute_metrics as compute_metrics_with_task, metric_to_watch, metric_watch_mode
from .base_lightning_model import BaseLightningModel
from .pretrained_transformer import PretrainedTransformer


logger = logging.getLogger(__name__)


def first_true_idx(tensor, dim, cumsum=None):
    # Takes a bool tensor and returns the idx of the first element that is True in a given dimension
    # Undefined return value iff all elements in the given dimension are False
    # The implementation is adapted from https://discuss.pytorch.org/t/first-nonzero-index/24769/9
    cumsum = cumsum if cumsum is not None else tensor.cumsum(dim)
    return ((cumsum == 1) & tensor).max(dim)[1]


def last_true_idx(tensor, dim, cumsum=None):
    # Takes a bool tensor and returns the idx of the last element that is True in a given dimension
    # Returns 0 if all elements in the given dimension are False
    cumsum = cumsum if cumsum is not None else tensor.cumsum(dim)
    return first_true_idx(cumsum == cumsum.max(dim)[0].unsqueeze(dim), dim)


class SemanticEncoder(BaseLightningModel):
    def __init__(self, args):
        super().__init__(args)

        args.task = args.task.lower()

        self.use_semantic_graph = args.formalism is not None
        self.formalism = args.formalism

        self.output_mode = output_modes[args.task]
        self.metric_to_watch = metric_to_watch[args.task]
        self.metric_watch_mode = metric_watch_mode[args.task]

        self.has_secondary_split = args.task in secondary_processors

        # transformer
        self.num_labels = tasks_num_labels.get(args.task)
        self.transformer = PretrainedTransformer(args, num_labels=self.num_labels)
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)

        assert args.activation in ACT2FN
        self.activation = ACT2FN[args.activation]

        # Data preparation; this must come after the initialization of self.transformer
        self.prepare_data()
        self.is_sentence_pair_task, self.relation2id, self.num_relations = pickle.load(open(self._metadata_file(), 'rb'))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_a_masks=None, sent_b_masks=None, graphs_a=None, gdata_a=None, graphs_b=None, gdata_b=None):
        """
        graphs_*: batched DGLGraph across sentences;
            bsz * DGLGraph
        g_data_*: dictinoary with values having shape:
            (bsz, ...)
        """
        raise NotImplementedError('This is an abstract class')

    def compute_loss(self, output_dict, labels):
        logits = output_dict["logits"]
        if self.output_mode == "classification":
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        elif self.output_mode == "regression":
            loss = F.mse_loss(logits.view(-1), labels.view(-1))
        return loss

    def compute_metrics(self, *args):
        return compute_metrics_with_task(self.args.task, *args)

    def pool_node_embeddings(self, last_layers, masks, gdata, batch_num_nodes):
        """
        Convert wordpiece embeddings into word (i.e. node) embeddings using the alignment in
        wpidx2graphid = gdata['wpidx2graphid']

        Parameters:
            g_data: dictinoary with values having shape:
                (bsz, ...)
            masks: (bsz, max_sent_pair_len)
            last_layers: (bsz, max_sent_pair_len, emb_dim)

        Returns:
            node_embs: (bsz, max_num_nodes, emb_dim)
            node_embeddings_mask: (bsz, max_num_nodes)
        """
        wpidx2graphid = gdata['wpidx2graphid']  # (bsz, max_sent_len, max_n_nodes)
        device = last_layers.device
        bsz, max_sent_len, max_n_nodes = wpidx2graphid.shape
        emb_dim = last_layers.shape[-1]
        assert max(batch_num_nodes) == wpidx2graphid.shape[-1]

        # the following logic happens to work if the graph is empty, in which case its sentence_end is guaranteed to be 1 (exclusive)
        masks_cumsum = masks.cumsum(1)
        sentence_starts = first_true_idx(masks, 1, masks_cumsum)
        sentence_ends = last_true_idx(masks, 1, masks_cumsum) + 1  # exclusive
        max_sentence_len = (sentence_ends - sentence_starts).max()

        # we're using a for loop here since only doing rolling across the batch dimension shouldn't be very expensive
        # that said, can we do it without a loop?
        rolled_last_layers = torch.stack([last_layer.roll(-sentence_start.item(), dims=0) for last_layer, sentence_start in zip(last_layers, sentence_starts)])
        segmented_last_layers = rolled_last_layers[:, :max_sentence_len, :]  # (bsz, max_sent_len, emb_dim)
        assert segmented_last_layers.shape[:2] == wpidx2graphid.shape[:2]

        # (bsz, max_sent_len, max_n_nodes, emb_dim)
        expanded_wpidx2graphid = wpidx2graphid.unsqueeze(-1).expand(-1, -1, -1, emb_dim)
        expanded_segmented_last_layers = segmented_last_layers.unsqueeze(2).expand(-1, -1, max_n_nodes, -1)

        # (bsz, max_n_nodes, emb_dim)
        node_embeddings = masked_mean(expanded_segmented_last_layers, expanded_wpidx2graphid, 1)

        node_embeddings = torch.where(expanded_wpidx2graphid.any(1), node_embeddings, torch.tensor(0., device=device))  # some nodes don't have corresponding wordpieces
        node_embeddings_mask = torch.arange(max(batch_num_nodes), device=device).expand(bsz, -1) < torch.tensor(batch_num_nodes, dtype=torch.long, device=device).unsqueeze(1)

        return node_embeddings, node_embeddings_mask

    def _check_input(self, *inputs):
        len_ = None
        for input_ in inputs:
            if input_ is None:
                continue
            if len_ is None:
                len_ = len(input_)
            assert len_ == len(input_)

    ###############################################################

    def _feature_file(self):
        return os.path.join(
            self.args.data_dir,
            f"features_{list(filter(None, self.args.model_name_or_path.split('/'))).pop()}"
            f"_{str(self.args.max_seq_length)}"
            f"{'_' + self.formalism if self.formalism is not None else ''}"
        )

    def _metadata_file(self):
        return os.path.join(
            self.args.data_dir,
            f"metadata_{list(filter(None, self.args.model_name_or_path.split('/'))).pop()}"
            f"_{str(self.args.max_seq_length)}"
            f"{'_' + self.formalism if self.formalism is not None else ''}"
        )

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.args

        cached_features_file = self._feature_file()
        cached_metadata_file = self._metadata_file()

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            return
        if getattr(self, 'data_features', None) is not None:
            return

        processor = processors[args.task]()
        if self.has_secondary_split:
            processor2 = secondary_processors[args.task]()

        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = {
            "train": processor.get_train_examples(args.data_dir),
            "dev": processor.get_dev_examples(args.data_dir),
            "test": processor.get_test_examples(args.data_dir),
        }
        if self.has_secondary_split:
            examples.update({
                "dev2": processor2.get_dev_examples(args.data_dir),
                "test2": processor2.get_test_examples(args.data_dir),
            })

        for k, v in examples.items():
            if isinstance(v, GeneratorType):
                examples[k] = list(v)

        is_sentence_pair_task = examples["train"][0].text_b is not None

        graphs = relation2id = num_relations = None
        if self.use_semantic_graph:
            graphs, relation2id, num_relations = get_graphs(
                args.data_dir,
                self.formalism,
                has_secondary_split=self.has_secondary_split,
            )

        features = {}
        for split in examples.keys():
            features[split] = convert_examples_to_features(
                examples[split],
                self.transformer.tokenizer,
                args.task,
                max_length=args.max_seq_length,
                graphs=graphs[split] if graphs is not None else None,
            )

        logger.info("Saving features into cached file %s", cached_features_file)
        pickle.dump(features, open(cached_features_file, 'wb'))
        pickle.dump((is_sentence_pair_task, relation2id, num_relations), open(cached_metadata_file, 'wb'))

    def load_dataset(self, mode, batch_size, shuffle=False):
        "Load datasets. Called after prepare data."

        if not hasattr(self, 'data_features'):
            logger.info('Loading feature files')
            self.data_features = pickle.load(open(self._feature_file(), 'rb'))
        features = self.data_features[mode]
        all_input_ids = [f.input_ids for f in features]
        all_attention_mask = [f.attention_mask for f in features]
        all_token_type_ids = [f.token_type_ids for f in features]
        all_labels = [f.label for f in features]
        all_sent_a_masks = [f.sent_a_mask for f in features]
        all_sent_b_masks = [f.sent_b_mask for f in features]
        all_graphs_a = [f.graph_a for f in features]
        all_graphs_b = [f.graph_b for f in features]

        dataloader = DataLoader(
            SemanticDataset(
                all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_sent_a_masks, all_sent_b_masks, all_graphs_a, all_graphs_b
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: SemanticDataset.collate_fn(
                batch,
                self.transformer.tokenizer.pad_token_id,
                self.transformer.tokenizer.pad_token_type_id,
                self.transformer.tokenizer.padding_side,
                self.output_mode,
            ),
        )

        # pytorch-lightning as of 0.8.0rc1 adds DistributedSampler to all dataloaders, but we only
        # want it to be added to the training dataloader. We achieve this by defaulting
        # `replace_sampler_ddp == False` and manually add the DistributedSampler to the training dataloader
        if mode == 'train' and self.args.gpus > 1:
            self.trainer.replace_sampler_ddp = True
            dataloader = self.trainer.auto_add_sampler(dataloader, True)
            self.trainer.replace_sampler_ddp = False

        return dataloader

    def transfer_batch_to_device(self, batch, device):
        # DGLGraph's .to method as of 0.4.3.post2 doesn't take the non_blocking arg
        # which pytorch-lightning passes. So we need to customize this behavior
        assert isinstance(batch, list)
        return [
            (
                [g.to(device) for g in e]
                if isinstance(e, list) and isinstance(e[0], DGLGraph)
                else move_data_to_device(e, device)
            )
            if e is not None else None
            for e in batch
        ]

    def put_graphs_to_cpu(self, graphs):
        cpu = torch.device('cpu')
        if graphs is None:
            return
        graphs.to(cpu)

    def _model_step(self, batch, batch_idx, compute_loss=False):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "sent_a_masks": batch[3],
            "sent_b_masks": batch[4],
        }
        if self.transformer.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2]
        labels = batch[5]
        device = inputs["input_ids"].device

        graphs_a_present = len(batch) > 6
        graphs_b_present = len(batch) > 8
        if graphs_a_present:
            inputs["graphs_a"] = batch[6]
            inputs["gdata_a"] = batch[7]
            for data in (inputs["graphs_a"].ndata, inputs["graphs_a"].edata):  # their devices aren't properly set by DDP which uses scatter instead of `.to`
                for k, v in data.items():
                    if v.device != device:
                        data[k] = v.to(device)
        if graphs_b_present:
            inputs["graphs_b"] = batch[8]
            inputs["gdata_b"] = batch[9]
            for data in (inputs["graphs_b"].ndata, inputs["graphs_b"].edata):  # their devices aren't properly set by DDP which uses scatter instead of `.to`
                for k, v in data.items():
                    if v.device != device:
                        data[k] = v.to(device)

        output_dict = self(**inputs)
        output = (output_dict, labels)
        if compute_loss:
            output += (self.compute_loss(output_dict, labels),)

        if graphs_a_present:
            self.put_graphs_to_cpu(batch[6])
        if graphs_b_present:
            self.put_graphs_to_cpu(batch[8])

        return output

    def recursively_detach(self, e):
        if isinstance(e, torch.Tensor):
            return e.detach().cpu()
        elif isinstance(e, (list, tuple)):
            return [self.recursively_detach(t) for t in e]
        elif isinstance(e, dict):
            return {k: self.recursively_detach(v) for k, v in e.items()}
        else:
            return e

    def training_step(self, batch, batch_idx):
        _, _, loss = self._model_step(batch, batch_idx, compute_loss=True)
        tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output_dict, labels, loss = self._model_step(batch, batch_idx, compute_loss=True)
        output_dict = self.recursively_detach(output_dict)
        labels = self.recursively_detach(labels)
        return {"val_loss": loss.detach().cpu(), "output_dict": output_dict, "target": labels}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output_dict, _ = self._model_step(batch, batch_idx)
        output_dict = self.recursively_detach(output_dict)
        return {"output_dict": output_dict}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        PretrainedTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            "--max_seq_length",
            default=None,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--task", type=str, required=True, help="The GLUE task to run",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )

        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )

        parser.add_argument(
            "--formalism",
            default=None,
            type=str,
            help="The semantic formalism to use.",
        )

        parser.add_argument(
            "--activation",
            default='relu',
            type=str,
            choices=ACT2FN.keys(),
            help=f"The activation function to use.",
        )

        return parser
