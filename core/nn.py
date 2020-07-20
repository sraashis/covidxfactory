import math

import torch
from torch import nn as nn

from classification import sep, iteration, save_predictions
from core.measurements import Avg, new_metrics
from core.utils import sep, plot_progress, NNDataLoader


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def save_checkpoint(cache, model):
    try:
        state_dict = model.module.state_dict()
    except:
        state_dict = model.state_dict()

    torch.save(state_dict, cache['log_dir'] + sep + cache['checkpoint'])


def load_checkpoint(cache, model):
    chk = torch.load(cache['log_dir'] + sep + cache['checkpoint'])
    try:
        model.module.load_state_dict(chk)
    except:
        model.load_state_dict(chk)


def save_if_better(cache, model, ep, score, direction):
    if (direction == 'maximize' and score > cache['best_score']) or (
            direction == 'minimize' and score < cache['best_score']):
        save_checkpoint(cache, model)
        cache['best_score'] = score
        cache['best_epoch'] = ep
        if cache['debug']:
            print(f"##### BEST! Model *** Saved *** : {cache['best_score']}")
    else:
        if cache['debug']:
            print(f"##### Not best: {score}, {cache['best_score']} in ep: {cache['best_epoch']}")


def evaluation(cache, nn, split_key=None, save_pred=False, dataset_list=None):
    nn['model'].eval()
    if cache.get('debug'):
        print(f'--- Running {split_key} ---')

    running_loss = Avg()
    eval_score = new_metrics(cache['num_class'])
    val_loaders = [NNDataLoader.new(shuffle=False, dataset=d, **cache) for d in dataset_list]
    with torch.no_grad():
        for loader in val_loaders:
            accumulator = [loader.dataset]
            score = new_metrics(cache['num_class'])
            for i, batch in enumerate(loader):
                it = iteration(cache, batch, nn)
                score.accumulate(it['scores'])
                running_loss.accumulate(it['loss'])
                accumulator.append([batch, it])
                if cache['debug'] and len(dataset_list) <= 1 and i % int(math.log(i + 1) + 1) == 0:
                    print(f"Itr:{i}/{len(loader)}, {it['loss'].average}, {it['scores'].scores()}")

            eval_score.accumulate(score)
            if cache['debug'] and len(dataset_list) > 1:
                print(f"{split_key}, {score.scores()}")
            if save_pred:
                save_predictions(cache, accumulator)

    if cache['debug']:
        print(f"{cache['experiment_id']} {split_key} scores: {eval_score.scores()}")
    return running_loss, eval_score


def train(cache, nn, dataset, val_dataset):
    train_loader = NNDataLoader.new(shuffle=True, dataset=dataset, **cache)
    for ep in range(1, cache['epochs'] + 1):
        nn['model'].train()
        running_loss = Avg()
        ep_score = new_metrics(cache['num_class'])
        for i, batch in enumerate(train_loader, 1):
            it = iteration(cache, batch, nn)
            running_loss.accumulate(it['loss'])
            ep_score.accumulate(it['scores'])
            if cache['debug'] and i % int(math.log(i + 1) + 1) == 0:
                print(f"Ep:{ep}/{cache['epochs']},Itr:{i}/{len(train_loader)},"
                      f"{it['loss'].average},{it['scores'].scores()}")

        cache['training_log'].append([running_loss.average, *ep_score.scores()])
        val_loss, val_score = evaluation(cache, nn, split_key='validation', dataset_list=[val_dataset])
        cache['validation_log'].append([val_loss.average, *val_score.scores()])
        save_if_better(cache, nn['model'], ep, val_score.f1, 'maximize')
        plot_progress(cache, plot_keys=['training_log', 'validation_log'])
