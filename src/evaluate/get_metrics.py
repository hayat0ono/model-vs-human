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
        assert 0 <= ood_accuracy_tmp <= 1, f'model: {model}, dataset{d.name}, ood accuracy: {ood_accuracy_tmp}'
        ood_accuracies.append(ood_accuracy_tmp)
        ood_accuracy[d.name] = ood_accuracy_tmp
    ood_accuracy['all'] = sum(ood_accuracies) / len(ood_accuracies)
    print(f"{model}, ood accuracy: {ood_accuracy['all']}")
    return ood_accuracy


def get_accuracy_difference(model, datasets, config):
    accuracy_difference = {}
    accuracy_differences = []
    for d in datasets:
        accuracy_difference_dataset = []
        df = get_experimental_data(d, [model.replace('_', '-')] + config.HUMAN_DATA_DICT[d.name])
        df_model = df.loc[df["subj"] == model]
        if len(d.experiments) == 0:
            for h in config.HUMAN_DATA_DICT[d.name]:
                df_human = df.loc[df["subj"] == h]
                assert len(df_model) == len(df_human), f'model: {model}, dataaset: {d.name}'
                accuracy_difference_tmp = a.SixteenClassAccuracyDifference().analysis(df_model, df_human)[
                    '16-class-accuracy-difference']
                accuracy_difference_dataset.append(accuracy_difference_tmp)
        else:
            for e in d.experiments:
                for c in e.data_conditions:
                    for h in config.HUMAN_DATA_DICT[d.name]:
                        df_model_cond = df_model.loc[(df_model["condition"] == c)]
                        df_human = df.loc[(df["condition"] == c) & (df["subj"] == h)]
                        assert len(df_model_cond) == len(df_human), f'model: {model}, dataaset: {d.name}'
                        accuracy_difference_tmp = a.SixteenClassAccuracyDifference().analysis(df_model_cond, df_human)[
                            '16-class-accuracy-difference']
                        accuracy_difference_dataset.append(accuracy_difference_tmp)
        accuracy_difference[d.name] = sum(accuracy_difference_dataset) / len(accuracy_difference_dataset)
        assert 0 <= accuracy_difference[d.name] <= 1, f'model: {model}, dataset: {d.name}, accuracy difference: {accuracy_difference[d.name]}'
        accuracy_differences.append(accuracy_difference[d.name])
    accuracy_difference['all'] = sum(accuracy_differences) / len(accuracy_differences)
    print(f"{model}, accuracy difference: {accuracy_difference['all']}")
    return accuracy_difference


def get_observed_consistency(model, datasets, config):
    observed_consistency = {}
    observed_consistencies = []
    for d in datasets:
        observed_consistency_dataset = []
        df = get_experimental_data(d, [model.replace('_', '-')] + config.HUMAN_DATA_DICT[d.name])
        df_model = df.loc[df["subj"] == model]
        if len(d.experiments) == 0:
            for h in config.HUMAN_DATA_DICT[d.name]:
                df_human = df.loc[df["subj"] == h]
                assert len(df_model) == len(df_human), f'model: {model}, dataaset: {d.name}'
                observed_consistency_tmp = a.ErrorConsistency().analysis(df_model, df_human)['observed-consistency']
                observed_consistency_dataset.append(observed_consistency_tmp)
        else:
            for e in d.experiments:
                for c in e.data_conditions:
                    for h in config.HUMAN_DATA_DICT[d.name]:
                        df_model_cond = df_model.loc[(df_model["condition"] == c)]
                        df_human = df.loc[(df["condition"] == c) & (df["subj"] == h)]
                        assert len(df_model_cond) == len(df_human), f'model: {model}, dataaset: {d.name}'
                        observed_consistency_tmp = a.ErrorConsistency().analysis(df_model_cond, df_human)['observed-consistency']
                        observed_consistency_dataset.append(observed_consistency_tmp)
        observed_consistency[d.name] = sum(observed_consistency_dataset) / len(observed_consistency_dataset)
        assert 0 <= observed_consistency[d.name] <= 1, f'model: {model}, dataset: {d.name}, observed consistency: {observed_consistency[d.name]}'
        observed_consistencies.append(observed_consistency[d.name])
    observed_consistency['all'] = sum(observed_consistencies) / len(observed_consistencies)
    print(f"{model}, observed consistency: {observed_consistency['all']}")
    return observed_consistency


def get_error_consistency(model, datasets, config):
    error_consistency = {}
    error_consistencies = []
    for d in datasets:
        error_consistency_dataset = []
        df = get_experimental_data(d, [model.replace('_', '-')] + config.HUMAN_DATA_DICT[d.name])
        df_model = df.loc[df["subj"] == model]
        if len(d.experiments) == 0:
            for h in config.HUMAN_DATA_DICT[d.name]:
                df_human = df.loc[df["subj"] == h]
                assert len(df_model) == len(df_human), f'model: {model}, dataaset: {d.name}'
                error_consistency_tmp = a.ErrorConsistency().analysis(df_model, df_human)['error-consistency']
                error_consistency_dataset.append(error_consistency_tmp)
        else:
            for e in d.experiments:
                for c in e.data_conditions:
                    for h in config.HUMAN_DATA_DICT[d.name]:
                        df_model_cond = df_model.loc[(df_model["condition"] == c)]
                        df_human = df.loc[(df["condition"] == c) & (df["subj"] == h)]
                        assert len(df_model_cond) == len(df_human), f'model: {model}, dataaset: {d.name}'
                        error_consistency_tmp = a.ErrorConsistency().analysis(df_model_cond, df_human)['error-consistency']
                        error_consistency_dataset.append(error_consistency_tmp)
        error_consistency[d.name] = sum(error_consistency_dataset) / len(error_consistency_dataset)
        assert -1 <= error_consistency[d.name] <= 1, f'model: {model}, dataset: {d.name}, error consistency: {error_consistency[d.name]}'
        error_consistencies.append(error_consistency[d.name])
    error_consistency['all'] = sum(error_consistencies) / len(error_consistencies)
    assert -1 <= error_consistency['all'] <= 1
    print(f"{model}, error consistency: {error_consistency['all']}")
    return error_consistency


def get_shape_bias(model, datasets, config):
    assert len(datasets) == 1
    ds = datasets[0]
    assert ds.name == "cue-conflict"
    df = get_experimental_data(ds, [model.replace('_', '-')] + config.HUMAN_DATA_DICT[ds.name])
    classes = df["category"].unique()
    df_selection = df.loc[(df["subj"] == model)]
    shape_bias = {}
    shape_biases = []
    for cl in classes:
        df_class_selection = df_selection.query("category == '{}'".format(cl))
        shape_bias_tmp = a.ShapeBias().analysis(df=df_class_selection)['shape-bias']
        assert 0 <= shape_bias_tmp <= 1
        shape_bias[cl] = shape_bias_tmp
        shape_biases.append(shape_bias_tmp)
    shape_bias['all'] = sum(shape_biases) / len(shape_biases)
    print(f"{model}, shape bias: {shape_bias['all']}")
    return shape_bias


def get_metrics(metrics, model, config):
    metric_to_datasets = {}
    for metric in metrics:
        metric_to_datasets[metric] = get_experiments(config.METRIC_TO_DATASET_LIST[metric])
    metric_dict = {}
    for metric in metrics:
        datasets = metric_to_datasets[metric]
        if metric == 'ood_accuracy':
            obtained_metric = get_ood_accuracy(model, datasets)
        elif metric == 'accuracy_difference':
            obtained_metric = get_accuracy_difference(model, datasets, config)
        elif metric == 'observed_consistency':
            obtained_metric = get_observed_consistency(model, datasets, config)
        elif metric == 'error_consistency':
            obtained_metric = get_error_consistency(model, datasets, config)
        elif metric == 'shape_bias':
            obtained_metric = get_shape_bias(model, datasets, config)
        else:
            assert False
        metric_dict[metric] = obtained_metric
    return metric_dict