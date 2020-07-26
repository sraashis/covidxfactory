#!/usr/bin/env python
# coding: utf-8


import os

import core.nn
from classification import init_cache, init_nn
from core.utils import check_previous_logs, get_args

sep = os.sep
import os

import runs
from core.measurements import new_metrics
import core.utils as tu
from classification import KernelDataset
import json

RUNS = [runs.V7CF_DATASET]
if __name__ == "__main__":
    params = get_args()
    for run in RUNS:
        """
        Getting base dataset directory as args makes easier to work with google colab.
        """
        for k, v in run.items():
            if 'dir' in k:
                run[k] = params['data_dir'] + sep + run[k]
                os.makedirs(run[k], exist_ok=True)

        global_score, global_cache = new_metrics(params['num_class']), {'test_scores': []}
        for split_file in os.listdir(run['split_dir']):
            cache = init_cache(params, run, experiment_id=split_file.split('.')[0])
            cache['log_dir'] = cache['log_dir'] + os.sep + cache['dataset_name']
            os.makedirs(cache['log_dir'], exist_ok=True)
            check_previous_logs(cache)

            split = json.loads(open(run['split_dir'] + sep + split_file).read())
            nn = init_nn(cache, init_weights=True)
            if cache['phase'] == 'train':
                train_dataset = KernelDataset(mode='train', limit=params['load_limit'])
                train_dataset.add(dataset_id=cache['experiment_id'], files=split['train'], debug=params['debug'], **run)

                val_dataset = KernelDataset(mode='eval', limit=params['load_limit'])
                val_dataset.add(dataset_id=cache['experiment_id'], files=split['validation'], debug=params['debug'],
                                **run)

                core.nn.train(cache, nn, train_dataset, val_dataset)
                tu.save_cache(cache)

            core.nn.load_checkpoint(cache, nn['model'])
            test_dataset_list = []
            if cache.get('load_sparse'):
                for f in split['test']:
                    if len(test_dataset_list) >= params['load_limit']:
                        break
                    test_dataset = KernelDataset(mode='eval', limit=params['load_limit'])
                    test_dataset.add(dataset_id=cache['experiment_id'], files=[f], debug=False, **run)
                    test_dataset_list.append(test_dataset)
                if params['debug']:
                    print(f'{len(test_dataset_list)} sparse dataset loaded.')
            else:
                test_dataset = KernelDataset(mode='eval', limit=params['load_limit'])
                test_dataset.add(dataset_id=cache['experiment_id'], files=split['test'], debug=params['debug'],
                                 **run)
                test_dataset_list.append(test_dataset)

            test_loss, test_score = core.nn.evaluation(cache, nn, split_key='test', save_pred=True,
                                                       dataset_list=test_dataset_list)
            global_score.accumulate(test_score)
            cache['test_score'].append([split_file] + test_score.scores())
            global_cache['global_test_scores'].append([split_file] + test_score.scores())
            tu.save_scores(cache, experiment_id=cache['experiment_id'], file_keys=['test_score'])
        tu.save_scores(global_cache, experiment_id='global', file_keys=['test_scores'])
