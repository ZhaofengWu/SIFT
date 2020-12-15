from transformers.data.metrics import acc_and_f1, glue_compute_metrics, simple_accuracy

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    metrics = glue_compute_metrics(task_name, preds, labels)
    if task_name in ('mrpc', 'qqp'):
        del metrics['acc_and_f1']
    elif task_name == 'sts-b':
        del metrics['corr']
    return metrics

metric_to_watch = {
    'cola': 'mcc',
    'sst-2': 'acc',
    'mrpc': 'acc',
    'sts-b': 'pearson',
    'qqp': 'acc',
    'mnli': 'acc1',  # the in-domain matched set
    'mnli-mm': 'acc',
    'qnli': 'acc',
    'rte': 'acc',
    'wnli': 'acc',
}

metric_watch_mode = {k: 'max' for k in metric_to_watch.keys()}
