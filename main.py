from ConfigurationFiles.ConfigurationUtils import get_configurations
from DataPreparation.executor import prepare_dataset
from Process.Train import train

CFG_FILE = "config.yml"

if __name__ == "__main__":
    cfg = get_configurations(CFG_FILE)

    if cfg["mode"] == "train":
        cfg = cfg["train"]
        if cfg["prepare_dataset"]:
            prepare_dataset()
        train(cfg)




