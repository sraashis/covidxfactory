import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tmf
from PIL import Image as IMG
from torch.optim import Adam

import core.nn
from core.imageutils import Image
from core.measurements import new_metrics, Avg
from core.utils import NNDataset
from models import get_model
import json
import cv2
import random

sep = os.sep


class KernelDataset(NNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_label = kw.get('label_getter')
        self.get_mask = kw.get('mask_getter')
        self.labels = None

    def load_index(self, map_id, file_id, file):
        if not self.labels:
            lbl = self.dmap[map_id]['label_dir'] + sep + os.listdir(self.dmap[map_id]['label_dir'])[0]
            self.labels = json.loads(open(lbl).read())
        _file = file.split('.')[0] + '.png'
        h, p, c, r = self.labels[file]
        self.indices.append([map_id, file_id, _file, [h, p, c, r]])

    def __getitem__(self, index):
        map_id, file_id, file, labels = self.indices[index]
        dt = self.dmap[map_id]
        try:
            img_obj = Image()
            img_obj.load(dt['data_dir'], file)
            img_obj.load_mask(dt['mask_dir'], dt['mask_getter'])
            img_obj.apply_clahe()

            if len(img_obj.array.shape) == 3:
                img_obj.array = img_obj.array[:, :, 0]

            num, comp, stats, centriod = cv2.connectedComponentsWithStats(img_obj.mask)
            crop = np.zeros((2, 320, 192), dtype=np.uint8)
            for i, (a, b, c, d, _) in enumerate(stats[1:]):
                img_obj.array[img_obj.mask == 0] = 0
                crop[i] = np.array(IMG.fromarray(img_obj.array).crop([a, b, a + c, b + d]).resize((192, 320)))
            tensor = crop / 255.0

            if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
                tensor = np.flip(tensor, 0)

            if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
                tensor = np.flip(tensor, 1)
            return {'indices': self.indices[index], 'input': tensor.copy(),
                    'label': np.array(labels)}
        except:
            pass

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.Resize((384, 384)), tmf.ToTensor()])


def init_cache(params, run, **kw):
    cache = {**kw}
    cache.update(**params)

    cache['dataset_name'] = run['data_dir'].split(sep)[1] + '_' + cache['which_model']
    cache['training_log'] = ['Loss,Precision,Recall,F1,Accuracy']
    cache['validation_log'] = ['Loss,Precision,Recall,F1,Accuracy']
    cache['test_score'] = ['Split,Precision,Recall,F1,Accuracy']

    if params['checkpoint'] is None:
        cache['checkpoint'] = cache['experiment_id'] + '.pt'

    cache['best_score'] = 0.0
    cache['best_epoch'] = 0
    return cache


def init_nn(cache, init_weights=False):
    if torch.cuda.is_available() and cache['num_gpu'] > 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = get_model(which=cache['which_model'], in_ch=2, r=cache['model_scale'])
    optim = Adam(model.parameters(), lr=cache['learning_rate'])

    if cache['debug']:
        torch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total Params:', torch_total_params)

    if device.type == 'cuda' and cache.get('num_gpu') > 1:
        model = torch.nn.DataParallel(model, list(range(cache.get('num_gpu'))))
    if init_weights:
        torch.manual_seed(cache['seed'])
        core.nn.initialize_weights(model)
    return {'device': device, 'model': model.to(device), 'optimizer': optim}


def multi_reg(multi, reg):
    multi = F.softmax(multi, 1)
    r = torch.cat([reg, reg ** 2, reg ** 3], 1).unsqueeze(1)
    return (multi * r).sum(2)


def iteration(cache, batch, nn):
    inputs = batch['input'].to(nn['device']).float()
    labels = batch['label'].to(nn['device']).float()

    if nn['model'].training:
        nn['optimizer'].zero_grad()

    if cache['which_model'] == 'multi_reg':
        loss, sc, out, pred = _iteration_multi_reg(cache, nn, inputs, labels)
    elif cache['which_model'] == 'multi':
        loss, sc, out, pred = _iteration_multi(cache, nn, inputs, labels)
    elif cache['which_model'] == "binary":
        loss, sc, out, pred = _iteration_binary(cache, nn, inputs, labels)
    else:
        raise ValueError(cache['which_mode'] + " is not valid.")

    if nn['model'].training:
        loss.backward()
        nn['optimizer'].step()

    avg = Avg()
    avg.add(loss.item(), len(inputs))

    return {'loss': avg, 'output': out, 'scores': sc, 'predictions': pred}


def _iteration_multi_reg(cache, nn, inputs, labels):
    multi, reg = nn['model'](inputs)

    reg_loss = F.mse_loss(reg.squeeze(), labels[:, 3:].squeeze())
    mr = multi_reg(multi, reg)
    mr_loss = F.cross_entropy(mr, labels[:, 2:3].squeeze().long())

    loss = (reg_loss + mr_loss) / 2

    out = F.softmax(mr, 1)
    _, pred = torch.max(out, 1)
    sc = new_metrics(cache['num_class'])
    sc.add(pred, labels[:, 2:3].squeeze())
    return loss, sc, out, pred


def _iteration_multi(cache, nn, inputs, labels):
    """
    Multilabel classification but only report COVID19 scores.
    """
    multi = nn['model'](inputs)
    loss = F.cross_entropy(multi, labels[:, 0:3].long())
    out = F.softmax(multi[:, :, 2:3].squeeze(), 1)
    _, pred = torch.max(out, 1)
    sc = new_metrics(cache['num_class'])
    sc.add(pred, labels[:, 2:3].squeeze())
    return loss, sc, out, pred


def _iteration_binary(cache, nn, inputs, labels):
    """
    Binary classification for COVID19.
    """
    out = nn['model'](inputs)
    loss = F.cross_entropy(out, labels[:, 2:3].squeeze().long())
    out = F.softmax(out, 1)
    _, pred = torch.max(out, 1)
    sc = new_metrics(cache['num_class'])
    sc.add(pred, labels[:, 2:3].squeeze())
    return loss, sc, out, pred


def save_predictions(cache, accumulator):
    pass
