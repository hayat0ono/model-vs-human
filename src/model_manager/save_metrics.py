import json


def save_metrics(model_root, model_name, metrics_dict):
    model_dir = model_root / model_name.replace("-", "/")
    for metric in metrics_dict:
        file_name = metric + '.json'
        with open(model_dir / file_name, 'w') as f:
            json.dump(metrics_dict[metric], f)