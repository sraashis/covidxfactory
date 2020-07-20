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
from models import DiskExcNet

sep = os.sep


class KernelDataset(NNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_label = kw.get('label_getter')
        self.get_mask = kw.get('mask_getter')

    def load_index(self, map_id, file_id, file):
        self.indices.append([map_id, file_id, file])

    def __getitem__(self, index):
        map_id, file_id, file = self.indices[index]
        dt = self.dmap[map_id]

        img_obj = Image()
        img_obj.load(dt['data_dir'], file)
        img_obj.load_ground_truth(dt['label_dir'], dt['label_getter'])
        img_obj.apply_clahe()
        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_obj.array = np.flip(img_obj.array, 0)
            img_obj.ground_truth = np.flip(img_obj.ground_truth, 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img_obj.array = np.flip(img_obj.array, 1)
            img_obj.ground_truth = np.flip(img_obj.ground_truth, 1)

        img_tensor = self.transforms(IMG.fromarray(img_obj.array))

        gt = img_obj.ground_truth
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]

        gt = self.transforms(IMG.fromarray(gt))
        gt[gt == 255] = 1
        return {'indices': self.indices[index], 'input': img_tensor, 'label': gt.squeeze()}

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.Resize((384, 384)), tmf.ToTensor()])


def init_cache(params, **kw):
    cache = {**kw}
    cache.update(**params)

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

    model = DiskExcNet(cache['num_channel'], cache['num_class'], r=cache['model_scale']).to(device)
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


def iteration(cache, batch, nn):
    inputs = batch['input'].to(nn['device']).float()
    labels = batch['label'].to(nn['device']).long()
    if nn['model'].training:
        nn['optimizer'].zero_grad()

    out = nn['model'](inputs)
    loss = F.cross_entropy(out, labels)
    out = F.log_softmax(out, 1)

    _, pred = torch.max(out, 1)
    sc = new_metrics(cache['num_class'])
    sc.add(pred, labels)

    if nn['model'].training:
        loss.backward()
        nn['optimizer'].step()

    avg = Avg()
    avg.add(loss.item(), len(inputs))

    return {'loss': avg, 'output': out, 'scores': sc, 'predictions': pred}


def save_predictions(cache, accumulator):
    dataset_name = list(accumulator[0].dmap.keys()).pop()
    file = accumulator[1][0]['indices'][2][0].split('.')[0]
    out = accumulator[1][1]['output']
    img = out[:, 1, :, :].cpu().numpy() * 255
    img = np.array(img.squeeze(), dtype=np.uint8)
    IMG.fromarray(img).save(cache['log_dir'] + sep + dataset_name + '_' + file + '.png')
