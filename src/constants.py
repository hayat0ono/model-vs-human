MODEL_ROOT = '/home/ono/PycharmProjects/proj_metamer_cornet/model/'


DEFAULT_DATASETS = ["edge",
                    "silhouette",
                    "cue-conflict",
                    "colour",
                    "contrast",
                    "high-pass",
                    "low-pass",
                    "phase-scrambling",
                    "power-equalisation",
                    "false-colour",
                    "rotation",
                    "eidolonI",
                    "eidolonII",
                    "eidolonIII",
                    "uniform-noise",
                    "sketch",
                    "stylized"]

DEFAULT_METRICS = ['ood_accuracy', 'accuracy_difference', 'observed_consistency', 'error_consistency', 'shape_bias']


METRIC_TO_DATASET_LIST = {
    'ood_accuracy': DEFAULT_DATASETS,
    'accuracy_difference': DEFAULT_DATASETS,
    'observed_consistency': DEFAULT_DATASETS,
    'error_consistency': DEFAULT_DATASETS,
    "shape-bias": ["cue-conflict"],
    }


HUMAN_DATA_DICT = {
    "colour": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "contrast": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "cue-conflict": ['subject-01', 'subject-02', 'subject-03', 'subject-04', 'subject-05', 'subject-06', 'subject-07', 'subject-08', 'subject-09', 'subject-10'],
    "edge": ['subject-01', 'subject-02', 'subject-03', 'subject-04', 'subject-05', 'subject-06', 'subject-07', 'subject-08', 'subject-09', 'subject-10'],
    "eidolonI": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "eidolonII": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "eidolonIII": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "false-colour": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "high-pass": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "low-pass": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "phase-scrambling": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "power-equalisation": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "rotation": ['subject-01', 'subject-02', 'subject-03', 'subject-04'],
    "silhouette": ['subject-01', 'subject-02', 'subject-03', 'subject-04', 'subject-05', 'subject-06', 'subject-07', 'subject-08', 'subject-09', 'subject-10'],
    "sketch" : ['subject-01', 'subject-02', 'subject-03', 'subject-04', 'subject-05', 'subject-06', 'subject-07'],
    "stylized": ['subject-01', 'subject-02', 'subject-03', 'subject-04', 'subject-05'],
    "uniform-noise": ['subject-01', 'subject-02', 'subject-03', 'subject-04']
}