import numpy as np
import torch

EPS = 10e-5
PR = 4


class Avg:
    def __init__(self, eps=EPS, pr=PR):
        self.value = 0.0
        self.count = 0.0
        self.eps = eps
        self.pr = pr

    def add(self, val, n=1):
        self.value += val * n
        self.count += n

    @property
    def average(self):
        return round(self.value / max(self.count, self.eps), self.pr)

    def reset(self):
        self.value = 0.0
        self.count = 0.0

    def accumulate(self, other):
        self.value += other.value
        self.count += other.count

    def scores(self):
        return self.average


class Prf1a:
    def __init__(self, eps=EPS, pr=PR):
        super().__init__()
        self.eps = eps
        self.pr = pr
        self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0

    def update(self, tn=0, fp=0, fn=0, tp=0):
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def add(self, pred, true):
        y_true = true.clone().int().view(1, -1).squeeze()
        y_pred = pred.clone().int().view(1, -1).squeeze()

        y_true[y_true == 255] = 1
        y_pred[y_pred == 255] = 1

        y_true = y_true * 2
        y_cases = y_true + y_pred
        self.tp += torch.sum(y_cases == 3).item()
        self.fp += torch.sum(y_cases == 1).item()
        self.tn += torch.sum(y_cases == 0).item()
        self.fn += torch.sum(y_cases == 2).item()

    def accumulate(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn

    def reset(self):
        self.tn, self.fp, self.fn, self.tp = [0] * 4

    @property
    def precision(self):
        p = self.tp / max(self.tp + self.fp, self.eps)
        return round(p, self.pr)

    @property
    def recall(self):
        r = self.tp / max(self.tp + self.fn, self.eps)
        return round(r, self.pr)

    @property
    def accuracy(self):
        a = (self.tp + self.tn) / \
            max(self.tp + self.fp + self.fn + self.tn, self.eps)
        return round(a, self.pr)

    @property
    def f1(self):
        return self.f_beta(beta=1)

    def f_beta(self, beta=1):
        f_beta = (1 + beta ** 2) * self.precision * self.recall / \
                 max(((beta ** 2) * self.precision) + self.recall, self.eps)
        return round(f_beta, self.pr)

    def prfa(self, beta=1):
        return [self.precision, self.recall, self.f_beta(beta=beta), self.accuracy]

    def scores(self, beta=1):
        return self.prfa(beta)

    @property
    def overlap(self):
        o = self.tp / max(self.tp + self.fp + self.fn, self.eps)
        return round(o, self.pr)


class ConfusionMatrix:
    """
    x-axis is predicted. y-axis is true lable.
    F1 score from average precision and recall is calculated
    """

    def __init__(self, num_classes=None, device='cpu', eps=EPS):
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes).float()
        self.device = device
        self.eps = eps

    def reset(self):
        self.matrix = torch.zeros(self.num_classes, self.num_classes).float()
        return self

    def update(self, matrix):
        self.matrix += matrix

    def accumulate(self, other):
        self.matrix += other.matrix
        return self

    def add(self, pred, true):
        pred = pred.clone().long().reshape(1, -1).squeeze()
        true = true.clone().long().reshape(1, -1).squeeze()
        self.matrix += torch.sparse.LongTensor(
            torch.stack([pred, true]).to(self.device),
            torch.ones_like(pred).long().to(self.device),
            torch.Size([self.num_classes, self.num_classes])).to_dense().to(self.device)

    def precision(self, average=True):
        precision = [0] * self.num_classes
        for i in range(self.num_classes):
            precision[i] = self.matrix[i, i] / max(torch.sum(self.matrix[:, i]), self.eps)
        precision = np.array(precision)
        return sum(precision) / self.num_classes if average else precision

    def recall(self, average=True):
        recall = [0] * self.num_classes
        for i in range(self.num_classes):
            recall[i] = self.matrix[i, i] / max(torch.sum(self.matrix[i, :]), self.eps)
        recall = np.array(recall)
        return sum(recall) / self.num_classes if average else recall

    def f1(self, average=True):
        f_1 = []
        precision = [self.precision(average)] if average else self.precision(average)
        recall = [self.recall(average)] if average else self.recall(average)
        for p, r in zip(precision, recall):
            f_1.append(2 * p * r / max(p + r, self.eps))
        f_1 = np.array(f_1)
        return f_1[0] if average else f_1

    def accuracy(self):
        return self.matrix.trace().item() / max(self.matrix.sum().item(), self.eps)

    def prfa(self):
        return [self.precision(True), self.recall(True), self.f1(True), self.accuracy()]

    def scores(self):
        return self.prfa()


def new_metrics(num_classes=2):
    if num_classes == 2:
        return Prf1a()
    elif num_classes > 2:
        return ConfusionMatrix(num_classes)
    elif num_classes == 0:
        return Avg()
    else:
        raise ValueError('Number of classes must be greater than 2')
