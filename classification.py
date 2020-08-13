import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tmf
from PIL import Image as IMG

from easytorch.utils.imageutils import Image
from easytorch.core.measurements import Avg, Prf1a
from easytorch.core.nn import ETTrainer, ETDataset
from models import get_model
import json

import cv2

sep = os.sep


class KernelDataset(ETDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_label = kw.get('label_getter')
        self.get_mask = kw.get('mask_getter')
        self.labels = None

    def load_index(self, map_id, file):
        if not self.labels:
            lbl = self.dataspecs[map_id]['label_dir'] + sep + os.listdir(self.dataspecs[map_id]['label_dir'])[0]
            self.labels = json.loads(open(lbl).read())
        _file = file.split('.')[0] + '.png'
        h, p, c, r = self.labels[file]
        self.indices.append([map_id, _file, [1 - h, p, c, r]])

    def __getitem__(self, index):
        map_id, file, labels = self.indices[index]
        dt = self.dataspecs[map_id]
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


def multi_reg(multi, reg, loss_eps=1):
    multi = F.softmax(multi, 1)
    # reg = 2 / (reg + loss_eps)
    # mr = multi * reg[..., None]
    return multi.sum(2)


class KernelTrainer(ETTrainer):
    def __init__(self, args):
        super().__init__(args)

    def _init_nn_model(self):
        self.nn['model'] = get_model(self.args['which_model'], self.args['num_channel'], r=self.args['model_scale'])

    def iteration(self, batch):
        inputs = batch['input'].to(self.nn['device']).float()
        labels = batch['label'].to(self.nn['device']).long()

        if self.args['which_model'] == 'multi_reg':
            loss, sc, out, pred = self._iteration_multi_reg(inputs, labels)
        elif self.args['which_model'] == 'multi':
            loss, sc, out, pred = self._iteration_multi(inputs, labels)
        elif self.args['which_model'] == "binary":
            loss, sc, out, pred = self._iteration_binary(inputs, labels)
        else:
            raise ValueError(self.args['which_mode'] + " is not valid.")

        avg = Avg()
        avg.add(loss.item(), len(inputs))

        return {'loss': loss, 'avg_loss': avg, 'output': out, 'scores': sc, 'predictions': pred}

    def _iteration_multi_reg(self, inputs, labels):
        multi, reg = self.nn['model'](inputs)
        mr = multi_reg(multi, reg)

        reg_loss = F.mse_loss(reg.squeeze(), labels[:, 3:].squeeze().float())
        multi_loss = F.cross_entropy(multi, labels[:, 0:3].long())
        mr_loss = F.cross_entropy(mr, labels[:, 2:3].squeeze().long())

        loss = reg_loss + multi_loss + mr_loss

        out = F.softmax(mr, 1)
        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, labels[:, 2:3].squeeze())
        return loss, sc, out, pred

    def _iteration_multi(self, inputs, labels):
        """
        Multilabel classification but only report COVID19 scores.
        """
        multi = self.nn['model'](inputs)
        loss = F.cross_entropy(multi, labels[:, 0:3].long())
        out = F.softmax(multi[:, :, 2:3].squeeze(), 1)
        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, labels[:, 2:3].squeeze())
        return loss, sc, out, pred

    def _iteration_binary(self, inputs, labels):
        """
        Binary classification for COVID19.
        """
        out = self.nn['model'](inputs)
        loss = F.cross_entropy(out, labels[:, 2:3].squeeze().long())
        out = F.softmax(out, 1)
        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, labels[:, 2:3].squeeze())
        return loss, sc, out, pred

    def new_metrics(self):
        return Prf1a()

    def save_predictions(self, dataset, its):
        pass

    def reset_dataset_cache(self):
        self.cache['global_test_score'] = []
        self.cache['monitor_metrics'] = 'f1'
        self.cache['score_direction'] = 'maximize'
        self.cache['log_dir'] = self.cache['log_dir'] + '_' + self.args['which_model']

    def reset_fold_cache(self):
        self.cache['training_log'] = ['Loss,Precision,Recall,F1,Accuracy']
        self.cache['validation_log'] = ['Loss,Precision,Recall,F1,Accuracy']
        self.cache['test_score'] = ['Split,Precision,Recall,F1,Accuracy']
        self.cache['best_score'] = 0.0
