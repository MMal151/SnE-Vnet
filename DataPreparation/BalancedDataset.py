import pandas as pd
from sklearn.model_selection import train_test_split

from DataPreparation.DataPrepUtils import generate_bins, get_voxel_count, sort_voxels
from DataPreparation.Dataset import Dataset
from Process.ProcessUtils import augmentation_cm
from Utils.CommonUtils import save_csv
from Utils.VisualisationUtils import plot_column

CLASS_NAME = "[DataPreparation/BalancedDataset]"


class BalancedDataset(Dataset):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.bin_range = cfg["balanced"]["bin_range"]
        self.save_ordered_set = cfg["balanced"]["save_ordered_set"]
        self.read_from_file = cfg["balanced"]["read_from_file"]
        self.visualise = cfg["balanced"]["visualise_dataset"]

    def generate_splits(self):

        if self.read_from_file and len(self.bin_range.split(",")) > 0:
            binned_voxels = pd.read_csv("DataFiles/ordered_set.csv")
            self.bin_range = [int(i) for i in self.bin_range.split(",")]
        else:
            vc = get_voxel_count(
                self.Y)  # Dictionary of voxels value of each lesion mask; Key: Lesion Path, Value: Voxel Count

            if self.bin_range == "":
                self.bin_range = generate_bins(vc)
            else:
                self.bin_range = [int(i) for i in self.bin_range.split(",")]

            binned_voxels = sort_voxels(self.X, vc, self.bin_range)

            if self.save_ordered_set:
                save_csv("DataFiles/", "ordered_set.csv", binned_voxels)

        df_train, df_test, df_valid = _train_test_split(binned_voxels, len(self.bin_range), self.test_ratio,
                                                        self.valid_ratio, self.seed)

        # Aligning data to match super class
        self.train_x, self.train_y = df_train['X'].tolist(), df_train['Y'].tolist()
        self.valid_x, self.valid_y = df_valid['X'].tolist(), df_valid['Y'].tolist()
        self.test_x, self.test_y = df_test['X'].tolist(), df_test['Y'].tolist()

        if self.do_patching:
            if not self.random_patches:
                self.patch_coords = super().generate_ordered_patches()

        super().save_csv()

        if self.visualise:
            plot_column(binned_voxels, 'Bin_Id', filename="DataFiles/ordered_set.png")
            plot_column(df_train, 'Bin_Id', filename="DataFiles/train.png")
            plot_column(df_test, 'Bin_Id', filename="DataFiles/test.png")
            plot_column(df_valid, 'Bin_Id', filename="DataFiles/valid.png")


def _train_test_split(df, total_bins, test_ratio, valid_ratio, seed):
    lgr = CLASS_NAME + "[_train_test_split()]"

    bin_masks = [df['Bin_Id'].isin([i]) for i in range(0, total_bins)]
    df_train, df_test, df_valid = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for i in bin_masks:
        if not df[i].empty:
            try:
                train, test = train_test_split(df[i], test_size=test_ratio, random_state=seed)
                train, valid = train_test_split(train, test_size=valid_ratio, random_state=seed)
            except ValueError:
                print(f"{lgr}: One of the generated train/test/valid set maybe empty.")

            if not train.empty:
                df_train = pd.concat([df_train, train])
            if not test.empty:
                df_test = pd.concat([df_test, test])
            if not valid.empty:
                df_valid = pd.concat([df_valid, valid])

    return df_train, df_test, df_valid
