import pandas as pd
import os

from modelvshuman.helper.plotting_helper import get_short_imagename, read_data
import modelvshuman.constants as c


def read_csv_files_from_directory(dir_path, participants):

    assert os.path.exists(dir_path)
    assert os.path.isdir(dir_path)

    df = pd.DataFrame()
    for f in sorted(os.listdir(dir_path)):
        for participant in participants:
            if f.endswith(".csv") and participant in f:
                df2 = read_data(os.path.join(dir_path, f))
                df2.columns = [c.lower() for c in df2.columns]
                df = pd.concat([df, df2])
    return df


def get_experimental_data(dataset, participants, print_name=False):
    if print_name:
        print(dataset.name)
    experiment_path = os.path.join(c.RAW_DATA_DIR, dataset.name)
    assert os.path.exists(experiment_path), experiment_path + " does not exist."

    df = read_csv_files_from_directory(experiment_path, participants)
    df.condition = df.condition.astype(str)

    for experiment in dataset.experiments:
        if not set(experiment.data_conditions).issubset(set(df.condition.unique())):
            print(set(experiment.data_conditions))
            print(set(df.condition.unique()))
            raise ValueError("Condition mismatch")

    df = df.copy()
    df["image_id"] = df["imagename"].apply(get_short_imagename)

    return df