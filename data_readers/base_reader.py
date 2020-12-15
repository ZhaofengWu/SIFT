from collections import Counter
import dataclasses
from dataclasses import dataclass
import json
import logging
import os
import pickle
from typing import Any, List, Optional, Tuple, Union

from dgl import DGLGraph
from rdflib import Graph as RDFGraph
import torch
from tqdm import tqdm

from .rdf2dgl import relations_in, rdf2dgl


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[Union[str, Tuple[RDFGraph, List], List[str]]] = None


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    sent_a_mask: Optional[List[int]] = None
    sent_b_mask: Optional[List[int]] = None
    graph_a: Optional[DGLGraph] = None
    graph_b: Optional[DGLGraph] = None
    label: Optional[Union[int, float, DGLGraph, List[int]]] = None
    metadata: Optional[Any] = None


def _flatten(l):
    return [e for subl in l for e in subl]


def all_rdf_to_dgl(all_split_names, all_rdf_graphs, all_metadata, bidirectional=True):
    assert len(all_split_names) == len(all_rdf_graphs) == len(all_metadata)

    relations = relations_in(_flatten(all_rdf_graphs))
    print(f'Relations count: {len(relations)}')
    relation2id = {rel: i for i, rel in enumerate(sorted(relations))}

    graphs = {}

    for split, rdf_graphs, metadata in zip(all_split_names, all_rdf_graphs, all_metadata):
        assert len(rdf_graphs) == len(metadata)
        graphs[split] = []
        for rdf_graph, mdata in zip(rdf_graphs, metadata):
            graph = rdf2dgl(
                rdf_graph, mdata, relation2id, bidirectional=bidirectional
            )
            graphs[split].append(graph)

    return graphs, relation2id, len(relations) * (2 if bidirectional else 1)


def get_graphs(
    data_dir,
    formalism,
    has_secondary_split=False,
):
    train_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'train.{formalism}.rdf'), 'rb'))
    dev_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'dev.{formalism}.rdf'), 'rb'))
    test_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'test.{formalism}.rdf'), 'rb'))
    if has_secondary_split:
        dev2_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'dev2.{formalism}.rdf'), 'rb'))
        test2_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'test2.{formalism}.rdf'), 'rb'))

    train_metadata = pickle.load(open(os.path.join(data_dir, f'train.{formalism}.metadata'), 'rb'))
    dev_metadata = pickle.load(open(os.path.join(data_dir, f'dev.{formalism}.metadata'), 'rb'))
    test_metadata = pickle.load(open(os.path.join(data_dir, f'test.{formalism}.metadata'), 'rb'))
    if has_secondary_split:
        dev2_metadata = pickle.load(open(os.path.join(data_dir, f'dev2.{formalism}.metadata'), 'rb'))
        test2_metadata = pickle.load(open(os.path.join(data_dir, f'test2.{formalism}.metadata'), 'rb'))

    assert len(train_rdf_graphs) == len(train_metadata)
    assert len(dev_rdf_graphs) == len(dev_metadata)
    assert len(test_rdf_graphs) == len(test_metadata)
    if has_secondary_split:
        assert len(dev2_rdf_graphs) == len(dev2_metadata)
        assert len(test2_rdf_graphs) == len(test2_metadata)

    all_split_names = ['train', 'dev', 'test']
    if has_secondary_split:
        all_split_names.extend(['dev2', 'test2'])
    all_rdf_graphs = [train_rdf_graphs, dev_rdf_graphs, test_rdf_graphs]
    if has_secondary_split:
        all_rdf_graphs.extend([dev2_rdf_graphs, test2_rdf_graphs])
    all_metadata = [train_metadata, dev_metadata, test_metadata]
    if has_secondary_split:
        all_metadata.extend([dev2_metadata, test2_metadata])

    return all_rdf_to_dgl(all_split_names, all_rdf_graphs, all_metadata)


def _spans_overlap(span_a, span_b):
    return not (span_a[1] <= span_b[0] or span_a[0] >= span_b[1])  # not no_overlap


def _assert_sorted(spans, allow_overlap=False):
    if spans is None:
        return
    for i, span in enumerate(spans):
        assert span[1] >= span[0]  # we allow empty spans
        if i > 0:
            if allow_overlap:
                assert span[0] >= spans[i - 1][1] or _spans_overlap(span, spans[i - 1])
            else:
                assert span[0] >= spans[i - 1][1]


def _determine_special_token_positions(special_token_indices):
    special_token_indices = sorted(special_token_indices)

    initial_special_tokens_start = special_token_indices[0]
    initial_special_tokens_end = -1
    for i, idx in enumerate(special_token_indices):
        if i != idx:
            initial_special_tokens_end = special_token_indices[i - 1] + 1
            break
    assert initial_special_tokens_end != -1

    final_special_tokens_start = -1
    final_special_tokens_end = special_token_indices[-1] + 1
    for i, idx in enumerate(special_token_indices[::-1]):
        if i + idx + 1 != final_special_tokens_end:
            final_special_tokens_start = special_token_indices[-i]
            break
    assert final_special_tokens_start != -1

    middle_special_tokens = [
        idx for idx in special_token_indices if not (
            initial_special_tokens_start <= idx < initial_special_tokens_end
        ) and not (
            final_special_tokens_start <= idx < final_special_tokens_end
        )
    ]
    if middle_special_tokens:
        middle_special_tokens_start = middle_special_tokens[0]
        middle_special_tokens_end = middle_special_tokens[-1] + 1
        assert middle_special_tokens == list(range(middle_special_tokens_start, middle_special_tokens_end))
    else:
        middle_special_tokens_start = middle_special_tokens_end = -1

    return initial_special_tokens_start, initial_special_tokens_end, middle_special_tokens_start, middle_special_tokens_end, final_special_tokens_start, final_special_tokens_end


def _split_offsets(wp_offsets, is_pair, special_token_indices):
    """
    [(0, 0), (0, 5), (5, 6), (0, 0), (0, 4), (0, 0)] ->
    [[(0, 5), (5, 6)], [(0, 4)]]

    This function assumes there is one and only one place of consecutive special tokens between the two sentences
    """
    (
        initial_special_tokens_start, initial_special_tokens_end, middle_special_tokens_start, middle_special_tokens_end, final_special_tokens_start, final_special_tokens_end
    ) = _determine_special_token_positions(special_token_indices)
    assert initial_special_tokens_start == 0 and final_special_tokens_end == len(wp_offsets)

    if not is_pair:
        assert middle_special_tokens_start == middle_special_tokens_end == -1
        offsets_a_span = (initial_special_tokens_end, final_special_tokens_start)
        offsets_b_span = (0, 0)
    else:
        offsets_a_span = (initial_special_tokens_end, middle_special_tokens_start)
        offsets_b_span = (middle_special_tokens_end, final_special_tokens_start)

    offsets_a = wp_offsets[offsets_a_span[0]:offsets_a_span[1]]
    offsets_b = wp_offsets[offsets_b_span[0]:offsets_b_span[1]]

    # Some tokenizers, e.g. RoBERTa, split some unicode characters in weird ways,
    # so we allow local non-sorted spans if they overlap
    _assert_sorted(offsets_a, allow_overlap=True)
    _assert_sorted(offsets_b, allow_overlap=True)

    return offsets_a, offsets_b, offsets_a_span, offsets_b_span


def _calc_wpidx2graphid(anchors, wp_offsets):
    """
    Parameters:
        anchors: List[Optional[Tuple[int, int]]]
        wp_offsets: List[Tuple[int, int]]

    Returns:
        List[List[bool]]
    """
    wpidx2graphid = [[False] * len(anchors) for _ in range(len(wp_offsets))]
    # There's probably an O(n) way to do this but the lists are usually short anyway
    for wp_idx, wp_span in enumerate(wp_offsets):
        for graph_id, node_span in enumerate(anchors):
            if node_span is not None and _spans_overlap(wp_span, node_span):
                wpidx2graphid[wp_idx][graph_id] = True

    return wpidx2graphid


def convert_examples_to_features(
    examples,
    tokenizer,
    task=None,
    max_length=None,
    graphs=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    from . import output_modes, processors  # avoid circular import
    processor = processors[task]()
    output_mode = output_modes[task]

    if graphs is not None:
        assert len(examples) * 2 == len(graphs) or len(examples) == len(graphs)

    for example in examples:
        example.text_a = example.text_a.strip()
        if example.text_b is not None:
            example.text_b = example.text_b.strip()

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) if example.text_b is not None else example.text_a for example in examples],
        max_length=max_length,
        return_offsets_mapping=True,
    )
    all_special_token_ids = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id}

    features = []
    for i, example in enumerate(examples):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding if k != "offset_mapping"}
        if "attention_mask" in inputs:
            inputs["attention_mask"] = [bool(m) for m in inputs["attention_mask"]]
        wp_offsets = batch_encoding["offset_mapping"][i]

        sent_a_mask = sent_b_mask = graph_a = graph_b = None
        if graphs is not None:
            special_token_indices = [i for i, input_id in enumerate(inputs["input_ids"]) if input_id in all_special_token_ids]
            assert len(special_token_indices) == tokenizer.num_special_tokens_to_add(pair=example.text_b is not None)
            wp_offsets_a, wp_offsets_b, offsets_a_span, offsets_b_span = _split_offsets(
                wp_offsets,
                is_pair=example.text_b is not None,
                special_token_indices=special_token_indices,
            )
            if graphs is not None:
                sent_a_mask = [offsets_a_span[0] <= i < offsets_a_span[1] for i in range(len(wp_offsets))]
                sent_b_mask = [offsets_b_span[0] <= i < offsets_b_span[1] for i in range(len(wp_offsets))]

        if example.label is None:
            label = None
        else:
            label_list = processor.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}
            if output_mode == "classification":
                label = label_map[example.label]
            else:
                label = float(example.label)

        if graphs is not None:
            to_enumerate = []
            if example.text_b is None:
                graph_a = graphs[i]
                to_enumerate.append((graph_a, wp_offsets_a))
            else:
                graph_a = graphs[i * 2]
                graph_b = graphs[i * 2 + 1]
                to_enumerate.extend([(graph_a, wp_offsets_a), (graph_b, wp_offsets_b)])

            for graph, wp_offsets in to_enumerate:
                if graph is None: continue
                anchors = [metadata.get('anchors') for metadata in graph.gdata['metadata']]
                wpidx2graphid = torch.tensor(_calc_wpidx2graphid(anchors, wp_offsets), dtype=torch.bool)  # (n_wp, n_nodes)
                graph.gdata['wpidx2graphid'] = wpidx2graphid # TODO: if we really want to have some fun we can make this a sparse tensor
                del graph.gdata['metadata']  # save memory

        feature = InputFeatures(**inputs, sent_a_mask=sent_a_mask, sent_b_mask=sent_b_mask, graph_a=graph_a, graph_b=graph_b, label=label)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features
