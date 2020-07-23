#!/usr/bin/env python
# coding: utf-8


import os

import core.nn
import runs
from classification import KernelDataset
from classification import init_cache, init_nn
from core.measurements import new_metrics
from core.utils import save_cache, save_scores, get_args, check_previous_logs

sep = os.sep

"""
Pooled version only supports pooling of just one fold each dataset at the moment.
"""

RUNS = [runs.DRIVE, runs.DRISTI]
if __name__ == "__main__":
    params = get_args()
    global_score = new_metrics(params['num_class'])
    cache = init_cache(params, experiment_id='pooled')
    cache['log_dir'] = cache['log_dir'] + '_pooled'
    os.makedirs(cache['log_dir'], exist_ok=True)
    check_previous_logs(cache)

    nn = init_nn(cache, init_weights=True)
    if params['phase'] == 'train':
        train_dataset = KernelDataset.pool(runs=RUNS, data_dir=cache['data_dir'], split_key='train',
                                           limit=params['load_limit'], debug=params['debug'])
        val_dataset = KernelDataset.pool(runs=RUNS, data_dir=cache['data_dir'], split_key='validation',
                                         limit=params['load_limit'], debug=params['debug'])
        core.nn.train(cache, nn, train_dataset, val_dataset)
        cache['runs'] = RUNS
        save_cache(cache)

    core.nn.load_checkpoint(cache, nn['model'])
    test_dataset_list = KernelDataset.pool(runs=RUNS, data_dir=cache['data_dir'], split_key='test',
                                           limit=params['load_limit'], sparse=cache.get('load_sparse'), debug=params['debug'])
    test_loss, test_score = core.nn.evaluation(cache, nn, split_key='test', save_pred=True,
                                               dataset_list=test_dataset_list)
    global_score.accumulate(test_score)
    cache['test_score'].append(['Global'] + global_score.prfa())
    save_scores(cache, file_keys=['test_score'])
