import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split

from DataPreparation.DataPrepUtils import get_nonempty_patches, get_patch_coordinates_3D
from Process.ProcessUtils import augmentation_cm
from Utils.CommonUtils import get_all_file_paths, get_all_possible_subdirs, remove_dirs, save_csv, str_to_tuple

CLASS_NAME = "[DataPreparation/Dataset]"


class Dataset:

    def __init__(self, cfg):
        self.train_ratio = float(cfg["train_ratio"])
        self.test_ratio = float(cfg["test_ratio"])
        self.valid_ratio = float(cfg["valid_ratio"])
        self.seed = cfg["seed"]

        assert (self.train_ratio + self.test_ratio == 1.0) and (self.valid_ratio < self.train_ratio / 2), \
            f"{CLASS_NAME}: Error: Dataset ratio not valid." \
            f"train_ratio: [{self.train_ratio}] ; test_ratio: [{self.test_ratio}] ; valid_ratio: [{self.valid_ratio}]"

        # --- List of all sample points ---#
        self.X = get_all_file_paths(cfg["input_paths"], cfg["scan_ext"])
        self.Y = get_all_file_paths(cfg["input_paths"], cfg["msk_ext"])
        self.x_ext = cfg["scan_ext"]
        self.y_ext = cfg["msk_ext"]

        # --- List of train, test and validation sets ---#
        self.train_x, self.train_y = None, None
        self.valid_x, self.valid_y = None, None
        self.test_x, self.test_y = None, None

        # --- Patching Configurations ---#
        self.do_patching = cfg["do_patching"]

        if cfg["do_patching"]:
            self.patch_shape = str_to_tuple(cfg["patch"]["shape"])
            self.random_patches = cfg["patch"]["random"]
            self.stride = cfg["patch"]["stride"]
            self.alw_empty_patches = cfg["patch"]["alw_empty_patches"]
            self.patch_coords = None

    def generate_splits(self):
        x, self.test_x, y, self.test_y = train_test_split(self.X, self.Y, test_size=self.test_ratio,
                                                          random_state=self.seed)

        self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(x, y, test_size=self.test_ratio,
                                                                                  random_state=self.seed)


        if self.do_patching:
            if not self.random_patches:
                self.patch_coords = self.generate_ordered_patches()

        self.save_csv()

    def generate_ordered_patches(self):
        patch_coords = {}  # List of patch coordinates w.r.t to image shape.
        train_patches, valid_patches, test_patches = [], [], []
        patches_dict = {}

        for i in self.train_y:
            msk = nib.load(i)
            if not msk.shape in patch_coords.keys():
                patch_coords[msk.shape] = get_patch_coordinates_3D(msk.shape, self.stride, self.patch_shape)
            patches = patch_coords[msk.shape]
            if not self.alw_empty_patches:
                patches = get_nonempty_patches(msk, patches, self.patch_shape)

            train_patches.append(patches)

        patches_dict['train'] = train_patches

        for i in self.valid_y:
            msk = nib.load(i)
            if not msk.shape in patch_coords.keys():
                patch_coords[msk.shape] = get_patch_coordinates_3D(msk.shape, self.stride, self.patch_shape)
            valid_patches.append(patch_coords[msk.shape])

        patches_dict['valid'] = valid_patches

        for i in self.test_y:
            msk = nib.load(i)
            if not msk.shape in patch_coords.keys():
                patch_coords[msk.shape] = get_patch_coordinates_3D(msk.shape, self.stride, self.patch_shape)
            test_patches.append(patch_coords[msk.shape])

        patches_dict['test'] = test_patches

        return patches_dict

    def save_csv(self):
        if self.do_patching:
            save_csv("DataFiles/", "train.csv",
                     pd.DataFrame({'X': self.train_x, 'Y': self.train_y, 'patches': self.patch_coords['train']}))
            save_csv("DataFiles/", "valid.csv",
                     pd.DataFrame({'X': self.valid_x, 'Y': self.valid_y, 'patches': self.patch_coords['valid']}))
            save_csv("DataFiles/", "test.csv",
                     pd.DataFrame({'X': self.test_x, 'Y': self.test_y, 'patches': self.patch_coords['test']}))
        else:
            save_csv("DataFiles/", "train.csv",
                     pd.DataFrame({'X': self.train_x, 'Y': self.train_y, 'patches': None}))
            save_csv("DataFiles/", "valid.csv",
                     pd.DataFrame({'X': self.valid_x, 'Y': self.valid_y, 'patches': None}))
            save_csv("DataFiles/", "test.csv",
                     pd.DataFrame({'X': self.test_x, 'Y': self.test_y, 'patches': None}))
