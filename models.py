import torch.nn.functional as F
from easytorch.utils.tensorutils import safe_concat
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, k=3):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class COVDNet(nn.Module):
    def __init__(self, num_channels, reduce_by=1):
        super(COVDNet, self).__init__()
        self.A1_ = ConvBlock(num_channels, int(64 / reduce_by))
        self.A2_ = ConvBlock(int(64 / reduce_by), int(128 / reduce_by))
        self.A3_ = ConvBlock(int(128 / reduce_by), int(256 / reduce_by))

        self.A_mid = ConvBlock(int(256 / reduce_by), int(512 / reduce_by))

        self.A3_up = nn.ConvTranspose2d(int(512 / reduce_by), int(256 / reduce_by), kernel_size=2, stride=2)
        self._A3 = ConvBlock(int(512 / reduce_by), int(256 / reduce_by))

        self.A2_up = nn.ConvTranspose2d(int(256 / reduce_by), int(128 / reduce_by), kernel_size=2, stride=2)
        self._A2 = ConvBlock(int(256 / reduce_by), int(128 / reduce_by))

        self.A1_up = nn.ConvTranspose2d(int(128 / reduce_by), int(64 / reduce_by), kernel_size=2, stride=2)
        self._A1 = ConvBlock(int(128 / reduce_by), int(64 / reduce_by))

        self.enc1 = ConvBlock(int(64 / reduce_by), int(128 / reduce_by), p=0)
        self.enc2 = ConvBlock(int(128 / reduce_by), int(256 / reduce_by), p=0)
        self.enc3 = ConvBlock(int(256 / reduce_by), int(256 / reduce_by), p=0)
        self.enc4 = ConvBlock(int(256 / reduce_by), int(512 / reduce_by), p=0)
        self.enc5 = ConvBlock(int(512 / reduce_by), int(512 / reduce_by), p=0)
        self.flat_size = int(512 / reduce_by) * 6 * 2

    def forward(self, x):
        a1_ = self.A1_(x)
        a1_dwn = F.max_pool2d(a1_, kernel_size=2, stride=2)

        a2_ = self.A2_(a1_dwn)
        a2_dwn = F.max_pool2d(a2_, kernel_size=2, stride=2)

        a3_ = self.A3_(a2_dwn)
        a3_dwn = F.max_pool2d(a3_, kernel_size=2, stride=2)

        a_mid = self.A_mid(a3_dwn)

        a3_up = self.A3_up(a_mid)
        _a3 = self._A3(safe_concat(a3_, a3_up))

        a2_up = self.A2_up(_a3)
        _a2 = self._A2(safe_concat(a2_, a2_up))

        a1_up = self.A1_up(_a2)
        _a1 = self._A1(safe_concat(a1_, a1_up))

        _a1 = F.max_pool2d(_a1, kernel_size=2, stride=2)
        _a1 = self.enc1(_a1)

        _a1 = F.max_pool2d(_a1, kernel_size=2, stride=2)
        _a1 = self.enc2(_a1)

        _a1 = F.max_pool2d(_a1, kernel_size=2, stride=2)
        _a1 = self.enc3(_a1)

        _a1 = F.max_pool2d(_a1, kernel_size=2, stride=2)
        _a1 = self.enc4(_a1)

        _a1 = F.max_pool2d(_a1, kernel_size=2, stride=2)
        _a1 = self.enc5(_a1)

        _a1 = _a1.view(-1, self.flat_size)
        return _a1


class MultiLabelModule(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.fc0m = nn.Linear(in_size, 512)
        self.fc0_bn = nn.BatchNorm1d(512)
        self.fc1m = nn.Linear(512, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2m = nn.Linear(256, 64)
        self.fc3m = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.fc0_bn(self.fc0m(x)))
        x = F.relu(self.fc1_bn(self.fc1m(x)))
        x = F.relu(self.fc2m(x))
        x = self.fc3m(x)
        return x.view(x.shape[0], 2, -1)


class BinaryLabelModule(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.fc0m = nn.Linear(in_size, 512)
        self.fc0_bn = nn.BatchNorm1d(512)
        self.fc1m = nn.Linear(512, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2m = nn.Linear(256, 64)
        self.fc3m = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc0_bn(self.fc0m(x)))
        x = F.relu(self.fc1_bn(self.fc1m(x)))
        x = F.relu(self.fc2m(x))
        x = self.fc3m(x)
        return x


class MultiLabel(nn.Module):
    def __init__(self, in_ch, r=8):
        super().__init__()
        self.encoder = COVDNet(num_channels=in_ch, reduce_by=r)
        self.multi = MultiLabelModule(self.encoder.flat_size)

    def forward(self, x):
        x = self.encoder(x)
        return self.multi(x)


class Binary(nn.Module):
    def __init__(self, in_ch, r=8):
        super().__init__()
        self.encoder = COVDNet(num_channels=in_ch, reduce_by=r)
        self.cls = BinaryLabelModule(self.encoder.flat_size)

    def forward(self, x):
        x = self.encoder(x)
        return self.cls(x)


def get_model(which, in_ch=2, r=4):
    if which == 'multi':
        return MultiLabel(in_ch, r)
    elif which == 'binary':
        return Binary(in_ch, r)
