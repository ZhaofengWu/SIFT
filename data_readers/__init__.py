from transformers import DataProcessor
from transformers import glue_output_modes, glue_processors, glue_tasks_num_labels

from .base_reader import get_graphs

output_modes = dict(glue_output_modes)

processors = dict(glue_processors)

tasks_num_labels = dict(glue_tasks_num_labels)

secondary_processors = {'mnli': processors['mnli-mm']}

# Inject functions into DataProcessor
DataProcessor.get_graphs = get_graphs
