import json
import os
import pickle
import sys

from rdflib import Graph, Literal
from tqdm import tqdm


def parse_mrp(obj, remove_isolate_nodes=True):
    graph = Graph()
    if 'nodes' not in obj or 'edges' not in obj:
        return graph, []

    connected_node_ids = set()
    for edge in obj['edges']:
        if edge['source'] != -1 and edge['target'] != -1:
            connected_node_ids.add(edge['source'])
            connected_node_ids.add(edge['target'])

    nodes = []
    graph_metadata = []
    nodeid2nodeidx = {}
    for node in obj['nodes']:
        node_id = node['id']
        if remove_isolate_nodes and node_id not in connected_node_ids:
            continue

        node_idx = len(nodes)
        assert node_id not in nodeid2nodeidx
        nodeid2nodeidx[node_id] = node_idx

        nodes.append(Literal(node_idx))

        metadata = {}
        if 'anchors' in node:
            anchors = node['anchors']
            start = min(anchor['from'] for anchor in anchors)
            end = max(anchor['to'] for anchor in anchors)
            metadata['anchors'] = [start, end]

        if 'properties' in node or 'values' in node:
            assert len(node['properties']) == len(node['values'])
            for key, value in zip(node['properties'], node['values']):
                metadata[key] = value

        graph_metadata.append(metadata)

    label2literal = {}
    for edge in obj['edges']:
        label = edge['label']
        if label not in label2literal:
            label2literal[label] = Literal(label)

        if edge['source'] != -1 and edge['target'] != -1:
            graph.add((nodes[nodeid2nodeidx[edge['source']]], label2literal[label], nodes[nodeid2nodeidx[edge['target']]]))
        else:
            print('WARNING: BAD EDGE')

    if len(graph) == 0:
        print('WARNING: EMPTY GRAPH')

    return graph, graph_metadata  # TODO: maybe we can inject metadata into the graph objects


def convert(input_file, graph_output_file, metadata_output_file):
    graphs = []
    all_graph_metadata = []

    with open(input_file) as f:
        num_lines = sum(1 for line in open(input_file))
        # In rare cases when a sentence is empty, mtool/main.py (see decode_che_et_al.sh) will remove it.
        # This happens when, for example, the original instance is something like "{id}\t{sent_a}\t\t" for sentence pair tasks
        # (this is present in the QQP training set).
        # Luckily we keep track of the IDs so we know which ones are missing,
        # so we add these sentences back in the form of empty graphs.
        idx_diff = 0
        for idx, line in tqdm(enumerate(f), total=num_lines):
            line = line.strip()
            json_obj = json.loads(line)

            while idx + idx_diff < int(json_obj['id']):
                graphs.append(Graph())
                all_graph_metadata.append([])
                idx_diff += 1

            graph, graph_metadata = parse_mrp(json_obj)
            graphs.append(graph)
            all_graph_metadata.append(graph_metadata)
        assert num_lines - 1 + idx_diff == int(json_obj['id'])

    pickle.dump(graphs, open(graph_output_file, 'wb'))
    pickle.dump(all_graph_metadata, open(metadata_output_file, 'wb'))


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
