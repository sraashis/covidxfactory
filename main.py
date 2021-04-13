import argparse
import os

from easytorch import EasyTorch, default_ap

from classification import KernelDataset, KernelTrainer

sep = os.sep

ap = argparse.ArgumentParser(parents=[default_ap], add_help=False)
ap.add_argument("-nch", "--num_channel", default=3, type=int, help="Number of channels of input image.")
ap.add_argument('-wm', '--which_model', default='multi', type=str, help='Which model to load.')
ap.add_argument('-sz', '--model_scale', default=1, type=int, help='Mode width scale')


# --------------------------------------------------------------------------------------------
def get_label(file_name):
    return file_name.split('.')[0] + '.png'


def get_mask(file_name):
    return file_name.split('.')[0] + '.png'


V7CF_DATASET = {
    'name': 'v7_lab_cf',
    'data_dir': 'v7_lab_cf' + sep + 'images',
    'mask_dir': 'v7_lab_cf' + sep + 'masks',
    'label_dir': 'v7_lab_cf' + sep + 'labels',
    'split_dir': 'v7_lab_cf' + sep + 'splits',
    'label_getter': get_label,
    'mask_getter': get_mask,
}
runner = EasyTorch([V7CF_DATASET], ap, dataset_dir='datasets', split_ratio=[0.6, 0.2, 0.2])

if __name__ == "__main__":
    runner.run(KernelTrainer, KernelDataset)
