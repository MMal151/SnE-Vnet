import os.path

import pandas as pd

from DataGenerators.NiftiGenerator import Nifti3DGenerator
from Utils.CommonUtils import is_valid_dir, is_valid_file

CLASS_NAME = "[Loader/DataLoader]"


# Returns train, valid and test generators
def load_data(cfg):
    lgr = CLASS_NAME + "[load_data()]"

    assert is_valid_dir(cfg["loader"]["csv_path"]), f"{lgr}: CSV directory is invalid. "

    train_csv = os.path.join(cfg["loader"]["csv_path"], "train.csv")
    test_csv = os.path.join(cfg["loader"]["csv_path"], "test.csv")
    valid_csv = os.path.join(cfg["loader"]["csv_path"], "valid.csv")

    assert is_valid_file(train_csv) and is_valid_file(test_csv) and is_valid_file(valid_csv), \
        f"{lgr}: One (or more) data files either doesn't exist or have invalid path." \
        f"train_csv: [{train_csv}] ; test_csv: [{test_csv}] ; valid_csv: [{valid_csv}]"

    train_gen = Nifti3DGenerator(cfg["data"], pd.read_csv(train_csv), mode="train")
    test_gen = Nifti3DGenerator(cfg["data"], pd.read_csv(test_csv), mode="test")
    valid_gen = Nifti3DGenerator(cfg["data"], pd.read_csv(valid_csv), mode="valid")

    return train_gen, test_gen, valid_gen
