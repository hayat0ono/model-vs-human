import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Evaluate
from modelvshuman import constants as c


def evaluate(models):
    datasets = c.DEFAULT_DATASETS
    models_not_evaluated = {}
    for model in models:
        datasets_tmp = []
        for dataset in datasets:
            file_names = os.listdir(os.path.join(c.PROJ_DIR, f'raw-data/{dataset}/'))
            if len([item for item in file_names if f'_{model.replace("_", "-")}_' in item]) == 0:
                datasets_tmp.append(dataset)
        if len(datasets_tmp) == 0:
            continue
        else:
            models_not_evaluated[model] = datasets_tmp
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    for model in models_not_evaluated.keys():
        datasets = models_not_evaluated[model]
        Evaluate()([model], datasets, **params)
