import torch
import torch.nn as nn
from tps.utils.resample2d_package.resample2d import Resample2d
import torch.nn.init as init
from torchvision.ops import DeformConv2d

affine_par = True

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNetMulti_ours(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.warp = DeNetBoth(num_classes)
        #self.warp = DeNetBoth(1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.sf_layer = self.get_score_fusion_layer(num_classes, n_frames=2)
        self.warp_bilinear = Resample2d(bilinear=True) # Propagation

    def get_score_fusion_layer(self, num_classes, n_frames=2):
        sf_layer = nn.Conv2d(num_classes * n_frames, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.zeros_(sf_layer.weight)
        nn.init.eye_(sf_layer.weight[:, :num_classes, :, :].squeeze(-1).squeeze(-1))
        return sf_layer

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, cf, kf, nf, device, source='src', warp=False, prov_cf=None, prov_cf_aux=None, prov_kf=None, prov_kf_aux=None):
        if prov_cf is None:
            cf = self.conv1(cf)
            cf = self.bn1(cf)
            cf = self.relu(cf)
            cf = self.maxpool(cf)
            cf = self.layer1(cf)
            cf = self.layer2(cf)
            cf_feat = self.layer3(cf)
            if self.multi_level:
                cf_aux = self.layer5(cf_feat)
            else:
                cf_aux = None
            cf1 = self.layer4(cf_feat)
            cf = self.layer6(cf1)
        else:
            cf = prov_cf
            cf_aux = prov_cf_aux

        if prov_kf is None:
            with torch.no_grad():
                kf = self.conv1(kf)
                kf = self.bn1(kf)
                kf = self.relu(kf)
                kf = self.maxpool(kf)
                kf = self.layer1(kf)
                kf = self.layer2(kf)
                kf_feat = self.layer3(kf)
                if self.multi_level:
                    kf_aux = self.layer5(kf_feat)
                else:
                    kf_aux = None
                kf1 = self.layer4(kf_feat)
                kf = self.layer6(kf1)
        else:
            kf = prov_kf
            kf_aux = prov_kf_aux


        kf_warp_to_cf = self.warp(cf, kf)
        kf_aux_warp_to_cf_aux = self.warp(cf_aux, kf_aux)
        pred_aux = self.sf_layer(torch.cat((cf_aux, kf_aux_warp_to_cf_aux), dim=1))
        pred = self.sf_layer(torch.cat((cf, kf_warp_to_cf), dim=1))

        if warp:
            kf_warp_to_cf = self.warp(cf, kf)
            kf_aux_warp_to_cf_aux = self.warp(cf_aux, kf_aux)
            pred_aux = self.sf_layer(torch.cat((cf_aux, kf_aux_warp_to_cf_aux), dim=1))
            pred = self.sf_layer(torch.cat((cf, kf_warp_to_cf), dim=1))
            return pred_aux, pred, cf_aux, cf, kf_aux, kf

        return pred_aux, pred, cf_aux, cf, kf_aux, kf

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_1x_lr_params_sf_layer(self):
        b = []
        b.append(self.sf_layer.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_1x_lr_params_sf_layer(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]

def get_accel_deeplab_v2(num_classes=19, multi_level=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    return model


class DeNetBoth(nn.Module):
    def __init__(self,num_features,bias=False):
        super(DeNetBoth, self).__init__()

        self.bottleneck = nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=1, bias=bias)
        # Offset Setting
        kernel_size = 3
        #deform_groups = 8
        # cs
        deform_groups = 3

        # mvss
        #deform_groups = 1

        out_channels = deform_groups * 3 * kernel_size**2
        bias=False
        # Offset Conv
        self.offset_conv1 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        self.offset_conv2 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        self.offset_conv3 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        self.offset_conv4 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)

        # Deform Conv
        self.deform1 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform2 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform3 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)
        self.deform4 = DeformConv2d(num_features, num_features, 3, padding=1, groups=deform_groups)

    def offset_gen(self, x):

        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return offset, mask

    # X1 is ref, warping X2 to 1
    def forward(self, X1,X2):
        print(X1.size())
        jomi
        X = torch.cat((X1.unsqueeze(1),X2.unsqueeze(1)),dim=1)
        batch_size, frame, channel, h1, w1 = X.size()

        #X = X.reshape(batch_size*frame, channel, h1, w1)
        #X = self.conv1(X)          # (B, num_features, H/2, W/2)
        #X = X.reshape(batch_size, frame, X.size(1), h1, w1)
        # print('-conv1-',X.shape)
        ## Burst Feature Alignment
        #batch_size, frame, channel, h1, w1 = X.size()

        ref = X[:,0:1,:,:,:]

        ref = torch.repeat_interleave(ref, frame, dim=1)
        # print('-alignment 3-',ref.shape)
        feat = torch.cat([ref, X], dim=2)
        # print('-alignment 4-',feat.shape)
        feat = feat.reshape(batch_size*frame, channel*2, h1, w1)
        # print('-alignment 5-',feat.shape)
        feat = self.bottleneck(feat)
        # print('-alignment 6-',feat.shape)

        offset1, mask1 = self.offset_gen(self.offset_conv1(feat))
        feat = self.deform1(feat, offset1, mask1)

        offset2, mask2 = self.offset_gen(self.offset_conv2(feat))
        feat = self.deform2(feat, offset2, mask2)
        # print('-alignment 7-',feat.shape)

        offset3, mask3 = self.offset_gen(self.offset_conv3(feat))
        feat = self.deform3(feat, offset3, mask3)

        offset4, mask4 = self.offset_gen(self.offset_conv4(feat))
        aligned_feat = self.deform4(feat, offset4, mask4)

        return aligned_feat[1:2]
