import argparse
from easytorch.utils.defaultargs import ap
import dataspecs as dspec

from easytorch import EasyTorch
from classification import KernelDataset, KernelTrainer

ap = argparse.ArgumentParser(parents=[ap], add_help=False)
ap.add_argument('-wm', '--which_model', default='multi', type=str, help='Which model to load.')

dataspecs = [dspec.V7CF_DATASET]
runner = EasyTorch(ap, dataspecs)

if __name__ == "__main__":
    runner.run(KernelDataset, KernelTrainer)
