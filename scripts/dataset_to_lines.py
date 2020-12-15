import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from tqdm import tqdm

from data_readers import processors

TRAIN_FILE = 'train.lines'
DEV_FILE = 'dev.lines'
TEST_FILE = 'test.lines'


def convert(task, input_dir, output_dir):
    if any(os.path.isfile(os.path.join(output_dir, filename)) for filename in (TRAIN_FILE, DEV_FILE, TEST_FILE)):
        raise ValueError('Output file already exists.')

    processor = processors[task]()
    train_examples = processor.get_train_examples(input_dir)
    dev_examples = processor.get_dev_examples(input_dir)
    test_examples = processor.get_test_examples(input_dir)

    for examples, filename in ((train_examples, TRAIN_FILE), (dev_examples, DEV_FILE), (test_examples, TEST_FILE)):
        with open(os.path.join(output_dir, filename), 'w') as f:
            for ex_index, example in enumerate(tqdm(examples)):
                if example.text_b is None:
                    f.write(f'{ex_index}\t{example.text_a}\n')
                else:
                    f.write(f'{ex_index * 2}\t{example.text_a}\n')
                    f.write(f'{ex_index * 2 + 1}\t{example.text_b}\n')


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
