import collections
import json
import sys


def convert(input_file, output_file):
    with open(input_file) as f:
        with open(output_file, 'w') as o:
            for line in f:
                obj = json.loads(line, object_pairs_hook=collections.OrderedDict)
                output = {
                    'id': obj['id'],
                    'version': 1.0,
                    'time': obj['time'],
                    'source': 'none',
                    'targets': ['dm', 'psd', 'eds', 'ucca', 'amr'],
                    'input': obj['input'],
                }
                o.write(json.dumps(output) + '\n')


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
