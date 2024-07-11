import ast
import logging
import os
import shutil
import random

CLASS_NAME = "[Utils/CommonUtils]"


def is_valid_dir(pth):
    return bool(pth) and os.path.exists(pth) and os.path.isdir(pth)


def is_valid_file(file_name):
    return bool(file_name) and os.path.exists(file_name)


def is_valid_str(in_str):
    return in_str is not None and len(in_str) > 0 and in_str.strip() != ""


def get_all_possible_files_paths(root: str, ext: str):
    lgr = CLASS_NAME + "[get_all_possible_files_paths()]"

    assert is_valid_dir(root), f"{lgr}: Invalid root path. Either it doesn't exist or is not a directory." \
                               f"Root: [{root}] \n Ext: [{ext}]"

    file_paths = set()
    for curr_dir, sub_dirs, files in os.walk(root):
        for f in files:
            if f.endswith(ext) and (not os.path.isdir(f)):
                file_paths.add(os.path.join(curr_dir, f))
    return sorted(file_paths)


def get_all_file_paths(root: str, ext: str = ".png", sep=","):
    lgr = CLASS_NAME + "[get_all_file_paths()]"

    assert bool(root) and bool(ext), f"{lgr}: Invalid input. root: [{root}] \n ext:[{ext}]"

    file_paths = []
    for i in root.split(sep):
        file_paths = file_paths + get_all_possible_files_paths(i, ext)

    return file_paths


# Saving a dataframe to a CSV File
def save_csv(path, file_name, df):
    lgr = CLASS_NAME + "[save_csv()]"

    if not is_valid_dir(path):
        os.makedirs(path)

    csv_file = os.path.join(path, file_name)

    if is_valid_file(csv_file):
        print(f"{lgr}: File [{csv_file}] already exists and will be overwritten.")

    df.to_csv(csv_file)


def str_to_tuple(strings, data_type=int):
    strings = strings.replace("(", "").replace(")", "")
    mapped_data = map(data_type, strings.split(","))
    return tuple(mapped_data)


def str_to_list(string):
    return list(ast.literal_eval(string))


def remove_dirs(path, incl=None):
    lgr = CLASS_NAME + "[remove_dirs()]"

    if type(path) == list:
        if incl is None:
            incl = ""
        files = []
        for i in path:
            if incl in i:
                logging.debug(f"{lgr}: Deleting directory {i}")
                shutil.rmtree(i, ignore_errors=True)
            else:
                files.append(i)
        return files
    else:
        logging.debug(f"{lgr}: Deleting directory {path}")
        shutil.rmtree(path)


# Returns all sub-dirs within a folder
# Input: root -> Path of the folder
#        mode -> full_path - Return physical path of the folder. Else just return the name of each subdirectory.
def get_all_possible_subdirs(root: str, mode=""):
    lgr = CLASS_NAME + "[get_all_possible_subdirs()]"

    if is_valid_dir(root):
        file_paths = set()
        for curr_dir, sub_dirs, _ in os.walk(root):
            for f in sub_dirs:
                if mode == "full_path":
                    file_paths.add(os.path.join(curr_dir, f))
                else:
                    file_paths.add(f)
        return sorted(file_paths)
    else:
        logging.warning(f"{lgr}: Invalid root path. Either it doesn't exist or is not a directory.")


# Generate random indexes in-between two ranges.
def get_random_index(min_idx, max_idx):
    i = random.randint(min_idx, max_idx)
    j = random.randint(min_idx, max_idx)

    while i == j:
        i, j = get_random_index(min_idx, max_idx)

    return i, j


