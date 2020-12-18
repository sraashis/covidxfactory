import argparse
from easytorch.etargs import ap
ap = argparse.ArgumentParser(parents=[ap], add_help=False)

import dataspecs as dspec

from easytorch import EasyTorch
from classification import KernelDataset, KernelTrainer

ap = argparse.ArgumentParser(parents=[ap], add_help=False)
ap.add_argument('-wm', '--which_model', default='multi', type=str, help='Which model to load.')

dataspecs = [dspec.V7CF_DATASET]
runner = EasyTorch(dataspecs, ap)

if __name__ == "__main__":
    runner.run(KernelDataset, KernelTrainer)
