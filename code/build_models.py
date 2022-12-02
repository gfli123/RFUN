import torch
import torch.nn.functional as F
import numpy as np
from option import opt
from torch import nn
from torch.nn import Conv2d, Sequential, Tanh, ReLU, Conv3d, BatchNorm3d, MaxPool2d, AvgPool2d, Upsample
from module_utils import Res_Block2D


class PFRB_gai(nn.Module):
    '''
    Progressive Fusion Residual Block
    Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations, ICCV 2019
    '''

    def __init__(self, num_fea=64, num_channel=3):
        super(PFRB_gai, self).__init__()
        self.nf = num_fea
        self.nc = num_channel
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv0 = nn.Sequential(*[nn.Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)) for _ in range(num_channel)])
        self.conv1 = nn.Conv2d(self.nf * num_channel, self.nf, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Sequential(*[nn.Conv2d(self.nf * 2, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)) for _ in range(num_channel)])

    def forward(self, x):
        x1 = [self.lrelu(self.conv0[i](x[i])) for i in range(self.nc)]
        merge = torch.cat(x1, 1)
        base = self.lrelu(self.conv1(merge))
        x2 = [torch.cat([base, i], 1) for i in x1]
        x2 = [self.lrelu(self.conv2[i](x2[i])) for i in range(self.nc)]
        img_ref = x[0] + x2[0]
        img_other = x[1] + x2[1]
        ht_con = x[2] + x2[2]

        return img_ref, img_other, ht_con


class Recurrent(nn.Module):
    def __init__(self, nf, hf, scale):
        super(Recurrent, self).__init__()
        self.nf = nf
        self.hf = hf
        self.scale = scale

        self.res = PFRB_gai(self.nf, 3)
        self.ht_conv_1 = Conv2d(self.nf * 3, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.res_dif = Res_Block2D(self.nf)
        self.merge1 = Conv2d(self.nf * 2, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, img_ref, img_other, ht_con):
        img_ref, img_other, ht_con = self.res([img_ref, img_other, ht_con])
        ht_con = self.lrelu(self.ht_conv_1(torch.cat((img_ref, img_other, ht_con), dim=1)))

        dif = img_ref - img_other
        fusion_dif = self.res_dif(dif)

        outputs = self.lrelu(self.merge1(torch.cat((ht_con, fusion_dif), dim=1)))

        return img_ref, img_other, ht_con, outputs


class Image_VSR(nn.Module):
    def __init__(self, nf=128, hf=128, num_b=3, num_inputs=3):
        super(Image_VSR, self).__init__()
        self.nf = nf
        self.hf = hf
        self.num_b = num_b
        self.num_inputs = num_inputs
        self.scale = opt.scale
        self.conv_ref = Conv2d(3, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_other = Conv2d(3 * 2, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.pro1 = Recurrent(self.nf, self.hf, self.scale)
        self.pro2 = Recurrent(self.nf, self.hf, self.scale)
        self.pro3 = Recurrent(self.nf, self.hf, self.scale)
        self.fusion = Conv2d(self.nf * 3, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.last = Conv2d(self.nf, 48, (3, 3), stride=(1, 1), padding=(1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, inputs):
        B, C, T, H, W = inputs.shape

        hr = []
        ht_con = torch.zeros((B, self.nf, H, W), dtype=torch.float, device=inputs.device)
        for i in range(T):
            in_group = generate_group(inputs, i, self.num_inputs, T)

            img_ref = in_group[:, :, 1, :, :]
            img_ref_bic = F.interpolate(img_ref, scale_factor=self.scale, mode='bicubic', align_corners=False)
            img_ref = self.lrelu(self.conv_ref(img_ref))
            img_other = self.lrelu(self.conv_other(torch.cat((in_group[:, :, 0, :, :], in_group[:, :, 2, :, :]), dim=1)))

            img_ref, img_other, ht_con, outputs1 = self.pro1(img_ref, img_other, ht_con)
            img_ref, img_other, ht_con, outputs2 = self.pro2(img_ref, img_other, ht_con)
            img_ref, img_other, ht_con, outputs3 = self.pro3(img_ref, img_other, ht_con)
            outputs = self.lrelu(self.fusion(torch.cat((outputs1, outputs2, outputs3), dim=1)))

            outputs = self.lrelu(self.last(outputs))
            outputs = F.pixel_shuffle(outputs, self.scale) + img_ref_bic
            hr.append(outputs)

        outputs = torch.stack(hr, dim=2)

        return outputs


def generate_group(inputs, idx, num_inputs=3, t=7):
    index = np.array([idx - num_inputs // 2 + i for i in range(num_inputs)])
    index = np.clip(index, 0, t - 1).tolist()
    outputs = inputs[:, :, index]
    return outputs


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, Conv2d):
            nn.init.xavier_normal_(m.weight, gain=1.)
            nn.init.zeros_(m.bias)
        if isinstance(m, Conv3d):
            nn.init.xavier_normal_(m.weight, gain=1.)
            nn.init.zeros_(m.bias)
