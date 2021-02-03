import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tmf
from PIL import Image as IMG
from easytorch import ETTrainer
from easytorch.data import ETDataset
from easytorch.vision.imageutils import Image

from models import get_model

sep = os.sep


class KernelDataset(ETDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None

    def load_index(self, map_id, file):
        """
        Load labels:
            Labl is a json file with
                key: image_name.json ->
                value: labels for [healthy{0, 1}, Pneumonia{0, 1}, covid19{0,1}, *] Where * is mulit-class label {0, 1, 2}
                for Healthy, Pneumonia, and covid respectively-This multi-label is not used in this work.
                 We only do multi-label and binary classification.
        """
        if not self.labels:
            lbl = self.dataspecs[map_id]['label_dir'] + sep + os.listdir(self.dataspecs[map_id]['label_dir'])[0]
            self.labels = json.loads(open(lbl).read())
        _file = file.split('.')[0] + '.json'
        if self.labels.get(_file):
            h, p, c, r = self.labels[_file]
            self.indices.append([map_id, file, [h, p, c, r]])

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
            tensor = self.transforms(IMG.fromarray(img_obj.array))
            return {'indices': self.indices[index], 'input': tensor,
                    'label': np.array(labels)}
        except:
            pass

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.Resize((192, 320)), tmf.RandomHorizontalFlip(), tmf.RandomVerticalFlip(), tmf.ToTensor()])


class KernelTrainer(ETTrainer):
    def __init__(self, args):
        super().__init__(args)

    def _init_nn_model(self):
        self.nn['model'] = get_model(self.args['which_model'], self.args['num_channel'], r=self.args['model_scale'])

    def iteration(self, batch):
        inputs = batch['input'].to(self.device['gpu']).float()
        labels = batch['label'].to(self.device['gpu']).long()

        if self.args['which_model'] == 'multi':
            loss, sc, out, pred = self._iteration_multi(inputs, labels)
        elif self.args['which_model'] == "binary":
            loss, sc, out, pred = self._iteration_binary(inputs, labels)
        else:
            raise ValueError(self.args['which_mode'] + " is not valid.")

        avg = self.new_averages()
        avg.add(loss.item(), len(inputs))

        return {'loss': loss, 'averages': avg, 'output': out, 'metrics': sc, 'predictions': pred}

    def _iteration_multi(self, inputs, labels):
        multi = self.nn['model'](inputs)
        loss = F.cross_entropy(multi, labels[:, 0:3].long())
        out = F.softmax(multi, 1)
        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred.squeeze(), labels[:, 0:3].squeeze())
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

    def init_experiment_cache(self):
        self.cache['log_dir'] = self.cache['log_dir'] + '_' + self.args['which_model']
