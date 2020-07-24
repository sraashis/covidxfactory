import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import safe_concat


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BasicConv2d, self).__init__()
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cout = nn.Conv2d(in_channels + 2 * out_channels, out_channels, kernel_size=3, padding=1, stride=1,
                              bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.bn1(self.conv_3x3(x))
        x1 = F.relu(x1, inplace=True)
        x2 = self.bn2(self.conv_1x1(x))
        x2 = F.relu(x2, inplace=True)
        out = self.bn_out(self.cout(torch.cat([x, x1, x2], 1)))
        return F.relu(out, inplace=True)


class MXPConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mxp_k=2, mxp_s=2, **kw):
        super(MXPConv2d, self).__init__()
        self.conv = BasicConv2d(in_channels, out_channels)
        self.mx_k = mxp_k
        self.mxp_s = mxp_s

    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=self.mx_k, stride=self.mxp_s)
        return self.conv(x)


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kw):
        super(UpConv2d, self).__init__()
        self.conv_up = nn.ConvTranspose2d(in_channels, out_channels, **kw)
        self.conv = BasicConv2d(out_channels, out_channels)

    def forward(self, x):
        x = self.conv_up(x)
        return self.conv(x)


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


class RegressionModule(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.fc0m = nn.Linear(in_size, 512)
        self.fc0_bn = nn.BatchNorm1d(512)
        self.fc1m = nn.Linear(512, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2m = nn.Linear(256, 64)
        self.fc3m = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc0_bn(self.fc0m(x)))
        x = F.relu(self.fc1_bn(self.fc1m(x)))
        x = F.relu(self.fc2m(x))
        x = self.fc3m(x)
        return x


class DiskExcNet(nn.Module):
    def __init__(self, in_ch, r):
        super(DiskExcNet, self).__init__()
        self.c1 = BasicConv2d(in_ch, r)

        self.c2 = MXPConv2d(r, 2 * r, kernel_size=3, padding=1)
        self.c3 = MXPConv2d(2 * r, 4 * r, kernel_size=3, padding=1)
        self.c4 = MXPConv2d(4 * r, 8 * r, kernel_size=3, padding=1)

        self.c5 = UpConv2d(8 * r, 4 * r, kernel_size=2, stride=2, padding=0)
        self.c6 = UpConv2d(4 * r, 2 * r, kernel_size=2, stride=2, padding=0)
        self.c7 = UpConv2d(2 * r, r, kernel_size=2, stride=2, padding=0)

        self.c8 = MXPConv2d(2 * r, 4 * r, kernel_size=3, padding=1)
        self.c9 = MXPConv2d(4 * r, 8 * r, kernel_size=3, padding=1)
        self.c10 = MXPConv2d(8 * r, 16 * r, kernel_size=3, padding=1)

        self.c11 = MXPConv2d(24 * r, 8 * r, kernel_size=3, padding=1)
        self.c12 = MXPConv2d(8 * r, 4 * r, kernel_size=3, padding=1)
        self.c13 = MXPConv2d(4 * r, r, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.c1(x)
        x = self.c2(x1)
        x = self.c3(x)
        x4 = self.c4(x)

        x = self.c5(x4)
        x = self.c6(x)
        x7 = self.c7(x)

        x = self.c8(safe_concat(x1, x7))
        x = self.c9(x)
        x10 = self.c10(x)

        x = self.c11(safe_concat(x4, x10))
        x = self.c12(x)
        x = self.c13(x)

        return x.view(x.size(0), -1)


class MultiLabel(nn.Module):
    def __init__(self, in_ch, r=8):
        super().__init__()
        self.encoder = DiskExcNet(in_ch=in_ch, r=r)
        self.multi = MultiLabelModule(r * 5 * 3)

    def forward(self, x):
        x = self.encoder(x)
        return self.multi(x)


class Regression(nn.Module):
    def __init__(self, in_ch, r=8):
        super().__init__()
        self.encoder = DiskExcNet(in_ch=in_ch, r=r)
        self.reg = RegressionModule(r * 5 * 3)

    def forward(self, x):
        x = self.encoder(x)
        return self.reg(x)


class MultiClassRegression(nn.Module):
    def __init__(self, in_ch=2, r=8):
        super().__init__()
        self.encoder = DiskExcNet(in_ch=in_ch, r=r)
        self.multi = MultiLabelModule(r * 5 * 3)
        self.reg = RegressionModule(r * 5 * 3)

    def forward(self, x):
        x = self.encoder(x)
        multi = self.multi(x)
        reg = self.reg(x)
        return multi, reg


def get_model(which, in_ch=2, r=8):
    if which == 'multi':
        return MultiLabel(in_ch, r)
    elif which == 'reg':
        return Regression(in_ch, r)
    elif which == 'multi_reg':
        return MultiClassRegression(in_ch, r)

# ls
# device = torch.device('cuda')
m = MultiClassRegression(2, 16)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(m)
print('Total Params:', torch_total_params)
#
#
# i = torch.randn(4, 2, 320, 192)

# multi, reg = m(i.to(device))
# multi = F.softmax(multi, 1)
# l1 = (reg * multi[:, 0, :]).sum(1)
# l2 = (reg / 2 * multi[:, 1, :]).sum(1)
# l = F.softmax(torch.cat([l1.unsqueeze(1), l2.unsqueeze(1)], 1), 1)
# print(l)
# pass
