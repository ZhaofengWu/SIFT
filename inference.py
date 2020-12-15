import argparse
import csv
import json
import os
import re
from types import SimpleNamespace

import pytorch_lightning as pl
import torch

from data_readers import output_modes, processors
from data_readers.base_reader import convert_examples_to_features, get_graphs
from models import RGCNSemanticEncoder


def load_pretrained_model(model_dir, map_to_cpu=False, **kwargs):
    training_args = SimpleNamespace(**json.load(open(os.path.join(model_dir, 'args.json'))))
    for k, v in kwargs.items():
        setattr(training_args, k, v)

    # pytorch-lightning has some bug that prevents us from directly doing `RGCNSemanticEncoder.load_from_checkpoint`
    if not map_to_cpu:
        checkpoint = torch.load(os.path.join(model_dir, 'best.ckpt'))
    else:
        checkpoint = torch.load(os.path.join(model_dir, 'best.ckpt'), map_location=torch.device('cpu'))

    # Backword compatability
    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[re.sub(r'^rgcns\.0\.', 'rgcn.', k)] = v
    checkpoint['state_dict'] = new_state_dict

    model = RGCNSemanticEncoder(training_args)
    model.load_state_dict(checkpoint['state_dict'])
    model.freeze()

    return model


def prepare_model(add_additional_args_fn=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        required=True,
        help="The directory of the pre-trained model.",
    )
    parser.add_argument(
        "--override_data_dir",
        default=None,
        type=str,
        help="If provided, ovreride the data directory. You want to use this if using the pretrained models from the authors.",
    )
    parser.add_argument("--gpus", type=int, default=None)
    if add_additional_args_fn is not None:
        add_additional_args_fn(parser)
    args = parser.parse_args()

    if args.gpus is None:
        args.gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if 'CUDA_VISIBLE_DEVICES' in os.environ else 0

    load_kwargs = dict(gpus=args.gpus)
    if args.override_data_dir is not None:
        load_kwargs['data_dir'] = args.override_data_dir
    model = load_pretrained_model(args.model_dir, map_to_cpu=args.gpus==0, **load_kwargs)

    train_params = dict(
        gpus=args.gpus,
        default_root_dir=args.model_dir,
    )

    trainer = pl.Trainer(**train_params)

    if add_additional_args_fn is not None:
        return trainer, model, args
    else:
        return trainer, model


def convert_predictions(predictions, task):
    if output_modes[task] == 'classification':
        processor = processors[task]()
        labels = processor.get_labels()
        predictions = [labels[pred] for pred in predictions]
    return predictions


def write_predictions(predictions, output_file):
    # This assumes that the test set maintains the original order and that the original indices
    # are [0, N)
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(enumerate(predictions))


def add_inference_data(parser):
    parser.add_argument(
        "--inference_task",
        default=None,
        type=str,
        help="The task of the inference data. If not provided, predict the original test set.",
    )
    parser.add_argument(
        "--inference_data_dir",
        default=None,
        type=str,
        help="The directory to the inference data. If not provided, predict the original test set.",
    )


def reset_test_dataloader(model, trainer, inference_task, inference_data_dir):
    processor = processors[inference_task]()
    test_examples = processor.get_test_examples(inference_data_dir)

    if model.use_semantic_graph:
        old_relation2id = model.relation2id

        model.args.task = inference_task
        model.args.data_dir = inference_data_dir

        graphs, model.relation2id, _ = get_graphs(inference_data_dir, model.formalism)
        graphs = graphs['test']

        # Re-map relations
        assert all(name in old_relation2id for name in model.relation2id.keys()), "Encountered semantic dependency relations not seen in training, aborting!"
        assert sorted(model.relation2id.values()) == list(range(max(model.relation2id.values()) + 1))
        assert sorted(model.relation2id.items(), key=lambda x: x[1]) == sorted(model.relation2id.items(), key=lambda x: x[0])
        relations_list = [rel for rel, id_ in sorted(model.relation2id.items(), key=lambda x: x[1])]
        transform_tensor = torch.tensor([old_relation2id[rel] for rel in relations_list])
        for graph in graphs:
            if len(graph.edges) == 0: continue

            assert (graph.edata['type'] < len(relations_list) * 2).all()
            graph.edata['type'] = transform_tensor[graph.edata['type'] % len(relations_list)] + graph.edata['type'] // len(relations_list) * len(relations_list)
    else:
        graphs = None

    data_features = {'test': convert_examples_to_features(test_examples, model.transformer.tokenizer, inference_task, model.args.max_seq_length, graphs)}
    model.data_features = data_features
    model.has_secondary_split = False  # if there's a separate inference task, we just do one split

    trainer.reset_test_dataloader(model)


def main():
    trainer, model, args = prepare_model(add_inference_data)
    if args.inference_task is not None:
        reset_test_dataloader(model, trainer, args.inference_task.lower(), args.inference_data_dir)
    trainer.test(model)

    if model.has_secondary_split:
        for i, predictions in enumerate(model.predictions):
            predictions = convert_predictions(predictions, model.args.task)
            write_predictions(predictions, os.path.join(args.model_dir, f'test_{i}_results.csv'))
    else:
        predictions = convert_predictions(model.predictions, model.args.task)
        write_predictions(predictions, os.path.join(args.model_dir, f'test_results.csv' if args.inference_task is None else f'test_{args.inference_task}_results.csv'))

if __name__ == "__main__":
    main()
