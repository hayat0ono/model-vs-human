import os


def get_metrics(metrics, model_root, model):
    metrics_not_evaluated = []
    model_files = os.listdir(model_root / model.replace('-', '/'))
    for metric in metrics:
        if not f'{metric}.json' in model_files:
            metrics_not_evaluated.append(metric)
    return metrics_not_evaluated