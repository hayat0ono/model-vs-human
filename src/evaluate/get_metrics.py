from modelvshuman.datasets.experiments import get_experiments
from modelvshuman.plotting import analyses as a
from .get_experimental_data import get_experimental_data


def get_ood_accuracy(model, datasets):
    ood_accuracy = {}
    ood_accuracies = []
    for d in datasets:
        df = get_experimental_data(d, [model.replace('_', '-')])
        df_selection = df.loc[(df["subj"] == model)]
        ood_accuracy_tmp = a.SixteenClassAccuracy().analysis(df_selection)['16-class-accuracy']
        ood_accuracies.append(ood_accuracy_tmp)
        ood_accuracy[d.name] = ood_accuracy_tmp
    ood_accuracy['all'] = sum(ood_accuracies) / len(ood_accuracies)
    return ood_accuracy


def get_accuracy_difference(model, datasets, config):
    accuracy_difference = {}
    accuracy_differences = []
    for d in datasets:
        df = get_experimental_data(d, [model.replace('_', '-')] + config.HUMAN_DATA_DICT[d.name])
        df_model = df.loc[df["subj"] == model]
        df_human = df.loc[df["subj"].isin(config.HUMAN_DATA_DICT[d.name])]
        accuracy_difference_tmp = a.SixteenClassAccuracyDifference().analysis(df_model, df_human)[
            '16-class-accuracy-difference']
        accuracy_difference[d.name] = accuracy_difference_tmp
        accuracy_differences.append(accuracy_difference_tmp)
    accuracy_difference['all'] = sum(accuracy_differences) / len(accuracy_differences)
    return accuracy_difference


def get_observed_consistency(model, datasets, config):
    observed_consistency = {}
    observed_consistencies = []
    for d in datasets:
        metrics_dataset = []
        df = get_experimental_data(d, [model.replace('_', '-')] + config.HUMAN_DATA_DICT[d.name])
        df_model = df.loc[df["subj"] == model]
        if len(d.experiments) == 0:
            for h in config.HUMAN_DATA_DICT[d.name]:
                df_human = df.loc[df["subj"] == h]
                assert len(df_model) == len(df_human)
                metric_tmp = a.ErrorConsistency().analysis(df_model, df_human)['observed_consistency']
                metrics_dataset.append(metric_tmp)
        else:
            for e in d.experiments:
                for c in e.data_conditions:
                    for h in config.HUMAN_DATA_DICT[d.name]:
                        df_model_cond = df_model.loc[(df_model["condition"] == c)]
                        df_human = df.loc[(df["condition"] == c) & (df["subj"] == h)]
                        assert len(df_model_cond) == len(df_human)
                        metric_tmp = a.ErrorConsistency().analysis(df_model_cond, df_human)['observed_consistency']
                        metrics_dataset.append(metric_tmp)
        observed_consistency[d.name] = sum(metrics_dataset) / len(metrics_dataset)
        observed_consistencies.append(observed_consistency[d.name])
    observed_consistency['all'] = sum(observed_consistencies) / len(observed_consistencies)
    return observed_consistency


def get_error_consistency(model, datasets, config):
    error_consistency = {}
    error_consistencies = []
    for d in datasets:
        metrics_dataset = []
        df = get_experimental_data(d, [model.replace('_', '-')] + config.HUMAN_DATA_DICT[d.name])
        df_model = df.loc[df["subj"] == model]
        if len(d.experiments) == 0:
            for h in config.HUMAN_DATA_DICT[d.name]:
                df_human = df.loc[df["subj"] == h]
                assert len(df_model) == len(df_human)
                metric_tmp = a.ErrorConsistency().analysis(df_model, df_human)['error_consistency']
                metrics_dataset.append(metric_tmp)
        else:
            for e in d.experiments:
                for c in e.data_conditions:
                    for h in config.HUMAN_DATA_DICT[d.name]:
                        df_model_cond = df_model.loc[(df_model["condition"] == c)]
                        df_human = df.loc[(df["condition"] == c) & (df["subj"] == h)]
                        assert len(df_model_cond) == len(df_human)
                        metric_tmp = a.ErrorConsistency().analysis(df_model_cond, df_human)['error_consistency']
                        metrics_dataset.append(metric_tmp)
        error_consistency[d.name] = sum(metrics_dataset) / len(metrics_dataset)
        error_consistencies.append(error_consistency[d.name])
    error_consistency['all'] = sum(error_consistencies) / len(error_consistencies)
    return error_consistency


def get_shape_bias(model, datasets, config):
    assert len(datasets) == 1
    ds = datasets[0]
    assert ds.name == "cue-conflict"
    df = get_experimental_data(ds, [model.replace('_', '-')] + config.HUMAN_DATA_DICT[d.name])
    classes = df["category"].unique()
    df_selection = df.loc[(df["subj"] == model)]
    shape_bias = {}
    shape_biases = []
    for cl in classes:
        df_class_selection = df_selection.query("category == '{}'".format(cl))
        shape_bias_tmp = a.ShapeBias().analysis(df=df_class_selection)['shape-bias']
        shape_bias[cl] = shape_bias_tmp
        shape_biases.append(shape_bias_tmp)
    shape_bias['all'] = sum(shape_biases) / len(shape_biases)
    return shape_bias


def get_metrics(metrics, model, config):
    metric_to_datasets = {}
    for metric in metrics:
        metric_to_datasets[metric] = get_experiments(config.METRICS_TO_DATASET_LIST[metric])
    dict_metrics = {}
    for metric in metrics:
        datasets = metric_to_datasets[metric]
        if metric == 'ood_accuracy':
            obtained_metric = get_ood_accuracy(model, datasets)
        elif metric == 'accuracy_difference':
            obtained_metric = get_accuracy_difference(model, datasets, config)
        elif metric == 'observed_difference':
            obtained_metric = get_observed_consistency(model, datasets, config)
        elif metric == 'error_consistency':
            obtained_metric = get_error_consistency(model, datasets, config)
        elif metric == 'shape_bias':
            obtained_metric = get_shape_bias(model, datasets, config)
        else:
            assert False
        dict_metrics[metric] = obtained_metric