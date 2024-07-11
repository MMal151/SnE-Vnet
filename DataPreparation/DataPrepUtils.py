import math
import nibabel as nib
import numpy as np
import pandas as pd

CLASS_NAME = "[DataPreparation/DataPrepUtils]"


# Get voxel count for each lesion file within the dataset.
def get_voxel_count(lesion_paths):
    lgr = CLASS_NAME + "[get_voxel_count()]"
    assert bool(lesion_paths) and len(lesion_paths) > 0, \
        f"{lgr}: Either given input path(s) or lesion file name is not valid. " \
        f"Lesion Paths: [{lesion_paths}]"

    data = {}
    for i in lesion_paths:
        mask = nib.load(i)
        data[i] = nib.imagestats.count_nonzero_voxels(mask)

    return data


def generate_bins(voxel_count):
    lgr = CLASS_NAME + "[generate_bins()]"

    assert bool(voxel_count) and len(voxel_count) > 0, f"{lgr}: Invalid voxel count." \
                                                       f" Voxel Count: [{voxel_count}]"

    total_bins = math.sqrt(len(voxel_count))
    bin_width = math.floor((max(voxel_count.values()) - min(voxel_count.values())) / total_bins)

    return [i for i in range(0, max(voxel_count.values()), bin_width)]


def get_bin(val, bins):
    for i in range(0, len(bins)):
        if (i + 1 < len(bins)) and bins[i] < val < bins[i + 1]:
            return i
    return len(bins) - 1


def sort_voxels(x, voxel_count, bins, append_max=False):
    lgr = CLASS_NAME + "[sort_voxels()]"
    assert len(bins) > 0, f"{lgr}: Invalid number of bins. Bins: [{bins}]"

    # Append max value as the range for the last bin.
    if append_max:
        bins.append(max(voxel_count.values()))

    binned_voxels = []

    for i, key in enumerate(voxel_count.keys()):
        binned_voxels.append(
            {'X': x[i], 'Y': key, 'Voxel_Count': voxel_count[key], 'Bin_Id': get_bin(voxel_count[key], bins)})

    return pd.DataFrame(binned_voxels, columns=['X', 'Y', 'Voxel_Count', 'Bin_Id'])


def get_patch_coordinates_3D(img_shape, stride, patch_shape):
    lgr = CLASS_NAME + "[get_patch_coordinates()]"

    if type(stride) is int:
        stride = (stride, stride, stride)

    assert len(img_shape) == 3, f"{lgr}: Invalid number of dimensions of shape. image_shape: [{img_shape}]"

    patch_coords = []
    for i in range(0, img_shape[0], stride[0]):
        for j in range(0, img_shape[1], stride[1]):
            for k in range(0, img_shape[2], stride[2]):
                if i + patch_shape[0] < img_shape[0] and \
                        j + patch_shape[1] < img_shape[1] and \
                        k + patch_shape[2] < img_shape[2]:
                    patch_coords.append((i, j, k))

    return patch_coords


def get_nonempty_patches(msk, patch_list, patch_shape):
    img = msk.get_fdata()
    non_empty_patches = []
    for coord in patch_list:
        (i, j, k) = coord
        if non_empty_patch(img[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]]):
            non_empty_patches.append(coord)
    return non_empty_patches


def non_empty_patch(patch):
    return len(np.unique(patch)) > 1
