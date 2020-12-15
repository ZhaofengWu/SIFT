import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


def _transpose(l):
    return list(zip(*l))


class SemanticDataset(Dataset):
    """When labels and/or graphs are not available, pass in a list of Nones"""

    def __init__(self, input_ids, attention_mask, token_type_ids, labels, sent_a_masks, sent_b_masks, graphs_a, graphs_b):
        assert len(input_ids) == len(attention_mask) == len(token_type_ids) == len(labels) == len(sent_a_masks) == len(sent_b_masks) == len(graphs_a) == len(graphs_b)

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.sent_a_masks = sent_a_masks
        self.sent_b_masks = sent_b_masks
        self.labels = labels
        self.graphs_a = graphs_a
        self.graphs_b = graphs_b

    def __getitem__(self, index):
        return [
            self.input_ids[index],
            self.attention_mask[index],
            self.token_type_ids[index],
            self.sent_a_masks[index],
            self.sent_b_masks[index],
            self.labels[index],
            self.graphs_a[index],
            self.graphs_b[index],
        ]

    def __len__(self):
        return len(self.input_ids)

    @staticmethod
    def _pad(example, pad_token, pad_token_segment_id, padding_side, max_length):
        input_ids, attention_mask, token_type_ids, sent_a_masks, sent_b_masks, label = example[:6]

        def _pad_sequence(sequence, padding_token, padding_length, padding_side):
            assert padding_side in {'left', 'right'}
            if sequence is None:
                return None
            if padding_side == 'left':
                return [padding_token] * padding_length + sequence
            else:
                return sequence + [padding_token] * padding_length

        padding_length = max_length - len(input_ids)
        input_ids = _pad_sequence(input_ids, pad_token, padding_length, padding_side)
        attention_mask = _pad_sequence(attention_mask, False, padding_length, padding_side)
        token_type_ids = _pad_sequence(token_type_ids, pad_token_segment_id, padding_length, padding_side)
        sent_a_masks = _pad_sequence(sent_a_masks, False, padding_length, padding_side)
        sent_b_masks = _pad_sequence(sent_b_masks, False, padding_length, padding_side)

        def _check_length(*seqs):
            len_ = None
            for seq in seqs:
                if seq is None:
                    continue
                if len_ is None:
                    len_ = len(seq)
                assert len(seq) == len_

        _check_length(input_ids, attention_mask, token_type_ids, sent_a_masks, sent_b_masks)

        example[:6] = input_ids, attention_mask, token_type_ids, sent_a_masks, sent_b_masks, label

    @staticmethod
    def _tensorize(example, output_mode):
        input_ids, attention_mask, token_type_ids, sent_a_masks, sent_b_masks, label = example[:6]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long) if token_type_ids is not None else None
        sent_a_masks = torch.tensor(sent_a_masks, dtype=torch.bool) if sent_a_masks is not None else None
        sent_b_masks = torch.tensor(sent_b_masks, dtype=torch.bool) if sent_b_masks is not None else None
        example[:5] = input_ids, attention_mask, token_type_ids, sent_a_masks, sent_b_masks
        label = torch.tensor(label, dtype=torch.long if output_mode != 'regression' else torch.float) if label is not None else None
        example[5] = label

    @staticmethod
    def _pad_and_stack_gdata(all_gdata, pad_value=False):
        max_shapes = {}
        for gdata in all_gdata:
            for k, tensor in gdata.items():
                if k not in max_shapes:
                    max_shapes[k] = list(tensor.shape)
                else:
                    max_shape = max_shapes[k]
                    for i, (max_, curr) in enumerate(zip(max_shape, tensor.shape)):
                        max_shape[i] = max(max_, curr)

        inner_output = {k: [] for k in max_shapes.keys()}
        for inner_idx, gdata in enumerate(all_gdata):
            for k, tensor in gdata.items():
                pad = []
                has_zero_dim = False
                for i, (max_, curr) in enumerate(zip(max_shapes[k][::-1], tensor.shape[::-1])):
                    pad.extend((0, max_ - curr))
                    if max_ == curr == 0:
                        has_zero_dim = True

                # There are rare cases where there exists a dim i that all gdata's dim i are 0
                # It happens when, e.g. all graphs in a batch are empty
                # F.pad will complain in that case
                if has_zero_dim:
                    inner_output[k].append(tensor.new_empty(max_shapes[k]))
                else:
                    inner_output[k].append(F.pad(tensor, pad, value=pad_value))

        return {k: torch.stack(tensors, 0) for k, tensors in inner_output.items()}

    @staticmethod
    def _prepare_graphs(graphs):
        batched_graphs = dgl.batch(graphs)
        gdata = [g.gdata for g in graphs]
        gdata = SemanticDataset._pad_and_stack_gdata(gdata)
        return batched_graphs, gdata

    @staticmethod
    def collate_fn(batch, pad_token, pad_token_segment_id, pad_on_left, output_mode):
        max_length = max(len(example[0]) for example in batch)
        for example in batch:
            SemanticDataset._pad(example, pad_token, pad_token_segment_id, pad_on_left, max_length)
            SemanticDataset._tensorize(example, output_mode)

        # input_ids, attention_mask, token_type_ids, sent_a_masks, sent_b_masks, labels, graphs_a, graphs_b
        n_fields_before_graphs = 6
        field_is_present = [field is not None for field in batch[0]]
        for example in batch:  # sanity check
            assert all(field_is_present[i] == (field is not None) for i, field in enumerate(example))

        # default_collate doesn't take None's, so we remove them and add them back
        not_none_output = default_collate(
            [
                [
                    field
                    for field, is_present in zip(e[:n_fields_before_graphs], field_is_present[:n_fields_before_graphs])
                    if is_present
                ]
                for e in batch
            ]
        )
        output = [None] * n_fields_before_graphs
        i = 0
        for j, is_present in enumerate(field_is_present[:n_fields_before_graphs]):
            if is_present:
                output[j] = not_none_output[i]
                i += 1

        graphs_a_present, graphs_b_present = field_is_present[n_fields_before_graphs:]
        if graphs_b_present:
            assert graphs_a_present

        if graphs_a_present:
            if not graphs_b_present:  # 1 graph
                graphs_a = [e[n_fields_before_graphs] for e in batch]
                output.extend(SemanticDataset._prepare_graphs(graphs_a))
            else:  # 2 graphs
                graphs_a, graphs_b = _transpose([e[n_fields_before_graphs:] for e in batch])
                output.extend(SemanticDataset._prepare_graphs(graphs_a))
                output.extend(SemanticDataset._prepare_graphs(graphs_b))

        return output
