import os

sep = os.sep
# --------------------------------------------------------------------------------------------
DRIVE = {
    'data_dir': 'DRIVE' + sep + 'images',
    'mask_dir': 'DRIVE' + sep + 'mask',
    'label_dir': 'DRIVE' + sep + 'OD_Segmentation',
    'split_dir': 'DRIVE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_gt.tif',
    'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
}

DRISTI = {
    'data_dir': 'DRISTI' + sep + 'Images',
    'label_dir': 'DRISTI' + sep + 'OD_Segmentation',
    'split_dir': 'DRISTI' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_ODsegSoftmap.png',
    'disk_center_getter': lambda file: file.split('.')[0] + '_diskCenter.txt'
}