import json
import os


def save_metrics(model_root, model_name, metrics_dict):
    model_dir = os.path.join(model_root, model_name.replace("-", "/"))
    for metric in metrics_dict:
        file_name = metric + '.json'
        with open(os.path.join(model_dir, file_name), 'w') as f:
            json.dump(metrics_dict[metric], f)