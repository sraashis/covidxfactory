import os

sep = os.sep
# --------------------------------------------------------------------------------------------
V7CF_DATASET = {
    'data_dir': 'v7_lab_cf' + sep + 'images',
    'mask_dir': 'v7_lab_cf' + sep + 'masks',
    'label_dir': 'v7_lab_cf' + sep + 'labels',
    'split_dir': 'v7_lab_cf' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '.png',
    'mask_getter': lambda file_name: file_name.split('.')[0] + '.png',
}
