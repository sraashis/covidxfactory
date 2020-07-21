import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

plt.switch_backend('agg')
plt.rcParams["figure.figsize"] = [16, 9]
import copy

sep = os.sep


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-nch", "--num_channel", default=3, type=int, help="Number of channels of input image.")
    ap.add_argument("-ncl", "--num_class", default=2, type=int, help="Number of output classes.")
    ap.add_argument("-b", "--batch_size", default=2, type=int, help="Mini batch size.")
    ap.add_argument('-ep', '--epochs', default=51, type=int, help='Number of epochs.')
    ap.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate.')
    ap.add_argument('-ngpu', '--num_gpu', default=2, type=int, help='Use GPU?')
    ap.add_argument('-pin', '--pin_memory', default=True, type=boolean_string, help='Pin Memory.')
    ap.add_argument('-nw', '--num_workers', default=2, type=int, help='Number of workers to work on data loading.')
    ap.add_argument('-p', '--phase', required=True, type=str, help='Phase of operation(train, validation, and test).')
    ap.add_argument('-data', '--data_dir', default='data', required=False, type=str, help='Root path to input Data.')
    ap.add_argument('-lim', '--load_limit', default=10e10, type=int, help='Data load limit')
    ap.add_argument('-log', '--log_dir', default='net_logs', type=str, help='Logging directory.')
    ap.add_argument('-chk', '--checkpoint', default=None, type=str, help='Logging directory.')
    ap.add_argument('-d', '--debug', default=True, type=boolean_string, help='Logging directory.')
    ap.add_argument('-s', '--seed', default=np.random.randint(1, 10e5), type=int, help='Seed')
    ap.add_argument('-f', '--force', default=False, type=boolean_string, help='Force')
    ap.add_argument('-r', '--model_scale', default=32, type=int, help='Force')
    return vars(ap.parse_args())


def create_k_fold_splits(files, k=0, save_to_dir=True, shuffle_files=True):
    from random import shuffle
    from itertools import chain
    import numpy as np

    if shuffle_files:
        shuffle(files)

    ix_splits = np.array_split(np.arange(len(files)), k)
    for i in range(len(ix_splits)):
        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = [ix for ix in np.arange(len(files)) if ix not in test_ix + val_ix]

        splits = {'train': [files[ix] for ix in train_ix],
                  'validation': [files[ix] for ix in val_ix],
                  'test': [files[ix] for ix in test_ix]}

        print('Valid:', set(files) - set(list(chain(*splits.values()))) == set([]))
        if save_to_dir:
            f = open('SPLIT_' + str(i) + '.json', "w")
            f.write(json.dumps(splits))
            f.close()
        else:
            return splits


def uniform_mix_two_lists(smaller, larger, shuffle=True):
    if shuffle:
        random.shuffle(smaller)
        random.shuffle(larger)

    len_smaller, len_larger = len(smaller), len(larger)

    accumulator = []
    while len(accumulator) < len_smaller + len_larger:
        try:
            for i in range(int(len_larger / len_smaller)):
                accumulator.append(larger.pop())
        except Exception:
            pass
        try:
            accumulator.append(smaller.pop())
        except Exception:
            pass

    return accumulator


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def plot_progress(cache, plot_keys=[], num_points=51):
    scaler = MinMaxScaler()
    for k in plot_keys:
        data = cache.get(k, [])

        if len(data) == 0:
            continue

        df = pd.DataFrame(data[1:], columns=data[0].split(','))

        if len(df) == 0:
            continue

        for col in df.columns:
            if max(df[col]) > 1:
                df[col] = scaler.fit_transform(df[[col]])

        rollin_window = max(df.shape[0] // num_points + 1, 3)
        rolling = df.rolling(rollin_window, min_periods=1).mean()
        ax = df.plot(x_compat=True, alpha=0.2, legend=0)
        rolling.plot(ax=ax, title=k.upper())

        plt.savefig(cache['log_dir'] + os.sep + f"{cache['experiment_id']}_{k}.png")
        plt.close('all')


def save_scores(cache, file_keys=[]):
    for fk in file_keys:
        with open(cache['log_dir'] + os.sep + f'{cache["experiment_id"]}_{fk}.csv', 'w') as file:
            for line in cache[fk] if any(isinstance(ln, list) for ln in cache[fk]) else [cache[fk]]:
                if isinstance(line, list):
                    file.write(','.join([str(s) for s in line]) + '\n')
                else:
                    file.write(f'{line}\n')


def find(fun, obj):
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if fun(v):
            obj[k] = ''
        elif isinstance(v, dict):
            find(fun, v)
        elif isinstance(v, list):
            for i in v:
                find(fun, i)


def save_cache(cache):
    with open(cache['log_dir'] + os.sep + f"{cache['experiment_id']}_log.json", 'w') as fp:
        log = copy.deepcopy(cache)
        find(callable, log)
        json.dump(log, fp)


def safe_concat(large, small):
    diff = np.array(large.shape) - np.array(small.shape)
    diffa = np.floor(diff / 2).astype(int)
    diffb = np.ceil(diff / 2).astype(int)

    t = None
    if len(large.shape) == 4:
        t = large[:, :, diffa[2]:large.shape[2] - diffb[2], diffa[3]:large.shape[3] - diffb[3]]
    elif len(large.shape) == 5:
        t = large[:, :, diffa[2]:large.shape[2] - diffb[2], diffa[3]:large.shape[3] - diffb[3],
            diffa[4]:large.shape[2] - diffb[4]]

    return torch.cat([t, small], 1)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def safe_collate(batch):
    return default_collate([b for b in batch if b])


class NNDataLoader(DataLoader):

    def __init__(self, **kw):
        super(NNDataLoader, self).__init__(**kw)

    @classmethod
    def new(cls, **kw):
        _kw = {
            'dataset': None,
            'batch_size': 1,
            'sampler': None,
            'shuffle': False,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': None
        }
        for k in _kw.keys():
            _kw[k] = kw.get(k, _kw.get(k))
        return cls(collate_fn=safe_collate, **_kw)


class NNDataset(Dataset):
    def __init__(self, mode='init', limit=float('inf'), **kw):
        self.mode = mode
        self.limit = limit
        self.indices = []
        self.dmap = {}

    def load_index(self, map_id, file_id, file):
        self.indices.append([map_id, file_id, file])

    def load_indices(self, map_id, files, **kw):
        for file_id, file in enumerate(files, 1):
            if len(self) >= self.limit:
                break
            self.load_index(map_id, file_id, file)

        if kw.get('debug', True):
            print(f'{map_id}, {self.mode}, {len(self)} Indices Loaded')

    def __getitem__(self, index):
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    @property
    def transforms(self):
        return None

    def add(self, dataset_id, files, debug=True, **kw):
        self.dmap[dataset_id] = kw
        self.load_indices(map_id=dataset_id, files=files, debug=debug)

    @classmethod
    def pool(cls, runs, data_dir='data', split_key=None, limit=float('inf'), sparse=False, debug=True):
        all_d = [] if sparse else cls(mode=split_key, limit=limit)
        for r_ in runs:
            """
            Getting base dataset directory as args makes easier to work with google colab.
            """
            r = {**r_}
            for k, v in r.items():
                if 'dir' in k:
                    r[k] = data_dir + sep + r[k]
            split = json.loads(open(r['split_dir'] + sep + os.listdir(r['split_dir'])[0]).read())
            if sparse:
                for file in split[split_key]:
                    if len(all_d) >= limit:
                        break
                    d = cls(mode=split_key)
                    d.add(dataset_id=r['data_dir'].split(sep)[1], files=[file], debug=False, **r)
                    all_d.append(d)
                if debug:
                    print(f'{len(all_d)} sparse dataset loaded.')
            else:
                all_d.add(dataset_id=r['data_dir'].split(sep)[1], files=split[split_key], debug=debug, **r)
        return all_d


def check_previous_logs(cache):
    if cache['force']:
        return
    i = 'y'
    if cache['phase'] == 'train':
        train_log = f"{cache['log_dir']}{sep}{cache['experiment_id']}_log.json"
        if os.path.exists(train_log):
            i = input(f"{train_log}' Exists. *** OVERRIDE *** [y/n]:")

    if cache['phase'] == 'test':
        test_log = f"{cache['log_dir']}{sep}{cache['experiment_id']}_test_scores.vsc"
        if os.path.exists(test_log):
            if os.path.exists(test_log):
                i = input(f"{test_log}' Exists. *** OVERRIDE *** [y/n]:")

    if i.lower() == 'n':
        raise FileExistsError(f' ##### {cache["log_dir"]} directory is not empty. #####')
