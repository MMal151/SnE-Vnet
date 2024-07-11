import nibabel as nib
import numpy as np
import tensorflow as tf

from Process.ProcessUtils import normalize_img
from Utils.CommonUtils import str_to_list, str_to_tuple

CLASS_NAME = "[DataGenerators/NiftiGenerator]"


class Nifti3DGenerator(tf.keras.utils.Sequence):
    def __init__(self, cfg, df, mode):

        self.x = df["X"].tolist()
        self.y = df["Y"].tolist()

        if df["patches"].isnull().all():
            self.patch_list = None
        else:
            self.patch_list = df["patches"].tolist()

        self.data_points = None

        assert len(self.x) == len(self.y), f"{CLASS_NAME}: Image set and label set doesn't have the same length."

        self.image_shape = str_to_tuple(cfg["input_shape"])
        self.batch_size = cfg["batch_size"]
        self.normalize = cfg["normalize"]
        self.shuffle = ["shuffle"]
        self.step_per_epoch = cfg["steps_per_epoch"][mode]

        self.init_dataset()

    def __len__(self):
        if self.step_per_epoch > 0:
            return self.step_per_epoch
        return int(np.floor(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data_points)

    def __getitem__(self, idx):
        return self.load_batch(idx)

    def load_batch(self, idx):

        image_shape = self.image_shape
        images, labels = np.zeros((self.batch_size, *image_shape, 1)), np.zeros((self.batch_size, *image_shape, 1))

        start_index = idx * self.batch_size
        end_index = start_index + self.batch_size

        batches = self.data_points[start_index: end_index]

        for i, batch in enumerate(batches):
            img = (nib.load(batch[0]).get_fdata())
            lbl = (nib.load(batch[1]).get_fdata())

            assert img.shape == lbl.shape, "Loaded image's shape != loaded label's shape"

            if self.normalize:
                img = normalize_img(img)

            if len(batch) == 3:
                (ax_1, ax_2, ax_3) = batch[2]
                img = img[ax_1: ax_1 + image_shape[0], ax_2: ax_2 + image_shape[1], ax_3: ax_3 + image_shape[2]]
                lbl = lbl[ax_1: ax_1 + image_shape[0], ax_2: ax_2 + image_shape[1], ax_3: ax_3 + image_shape[2]]

            images[i, :, :, :, :] = img.reshape((*image_shape, 1)).astype(np.float32)
            labels[i, :, :, :, :] = lbl.reshape((*image_shape, 1)).astype(np.float32)

        return images, labels

    def init_dataset(self):
        lgr = CLASS_NAME + "[init_data()]"

        if bool(self.patch_list) and len(self.patch_list) == len(self.x):
            for i in range(len(self.patch_list)):
                patches = str_to_list(self.patch_list[i])

                if self.data_points is None:
                    self.data_points = list(zip(np.repeat(self.x[i], len(patches)), np.repeat(self.y[i], len(patches)), patches))
                else:
                    self.data_points = self.data_points + list(zip(np.repeat(self.x[i], len(patches)), np.repeat(self.y[i], len(patches)), patches))
        else:
            print(f"{lgr}: Either patching is not enabled OR "
                  f"the number of instances in patch_list doesn't match the length of image/label set.")

            self.data_points = list(zip(self.x, self.y))

        self.on_epoch_end()
