from ConfigurationFiles.ConfigurationUtils import DATAPREP_CFG, get_configurations
from DataPreparation.BalancedDataset import BalancedDataset
from DataPreparation.Dataset import Dataset

CLASS_NAME = "[DataPreparation/executor]"


def prepare_dataset():
    cfg = get_configurations(DATAPREP_CFG)
    lgr = CLASS_NAME + "[execute()]"

    # Generate train/test/valid Splits
    if cfg["mode"].lower() == "gen_splits":
        if cfg["split"]["type"].lower() == "balanced":
            BalancedDataset(cfg["split"]).generate_splits()
        else:
            Dataset(cfg["split"]).generate_splits()


if __name__ == "__main__":
    DATAPREP_CFG = "..//" + DATAPREP_CFG
    prepare_dataset()
