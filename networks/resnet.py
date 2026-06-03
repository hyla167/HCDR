"""ResNet backbones and projection/prototype heads.

A single ``SupConResNet`` covers both the CCLIS baseline and the HCDR
extension.  When ``opt.use_hcdr`` is ``True`` the model additionally builds a
patch projection head and a set of patch-level prototypes, and its ``forward``
can return the spatial feature map (before global pooling) together with the
projected patch tokens and their prototype logits.
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        return (out, preact) if self.is_last else out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        return (out, preact) if self.is_last else out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init the last BN of each residual branch (https://arxiv.org/abs/1706.02677)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100, return_spatial=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # HCDR: spatial feature map BEFORE global pooling -> [B, 512, H, W]
        spatial_map = out if return_spatial else None
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if return_spatial:
            return out, spatial_map
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


def _build_head(head, dim_in, feat_dim):
    if head == 'linear':
        return nn.Linear(dim_in, feat_dim)
    if head == 'mlp':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
    raise NotImplementedError('head not supported: {}'.format(head))


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d via BatchNorm2d, for SyncBN purposes."""

    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        return x.view(-1, self.dim)


class SupConResNet(nn.Module):
    """Backbone + projection head + prototypes (+ optional HCDR patch heads).

    With ``opt.use_hcdr`` enabled, ``forward(x, return_spatial=True)`` returns
    ``(global_feat, global_proto_logits, raw_spatial_map, patch_feat,
    patch_proto_logits)`` where ``patch_feat`` is ``[B, H*W, feat_dim]`` and
    ``patch_proto_logits`` is ``[B, H*W, n_cls]``.
    """

    def __init__(self, name='resnet50', head='mlp', feat_dim=128, opt=None):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.head = _build_head(head, dim_in, feat_dim)

        self.prototypes = None
        if opt is not None:
            self.prototypes = nn.Linear(feat_dim, opt.n_cls, bias=False)

        # ---- HCDR additions -------------------------------------------------
        self.use_hcdr = bool(opt is not None and getattr(opt, 'use_hcdr', False))
        self.patch_head = None
        self.patch_prototypes = None
        if self.use_hcdr:
            self.patch_head = _build_head(head, dim_in, feat_dim)
            if opt is not None:
                self.patch_prototypes = nn.Linear(feat_dim, opt.n_cls, bias=False)

    def reinit_head(self):
        for layer in self.head.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, norm=True, return_spatial=False):
        want_spatial = return_spatial and self.use_hcdr
        if want_spatial:
            encoded, spatial_map = self.encoder(x, return_spatial=True)
        else:
            encoded = self.encoder(x)

        feat = self.head(encoded)
        if norm:
            feat = F.normalize(feat, dim=1)

        if self.prototypes is None:
            return feat

        global_logits = self.prototypes(feat).T  # [n_cls, B]
        if not want_spatial:
            return feat, global_logits

        # ---- HCDR patch branch ---------------------------------------------
        B, C, H, W = spatial_map.shape
        patches_raw = spatial_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
        patch_feat = self.patch_head(patches_raw)
        if norm:
            patch_feat = F.normalize(patch_feat, dim=-1)
        patch_proto_logits = self.patch_prototypes(patch_feat)
        return feat, global_logits, spatial_map, patch_feat, patch_proto_logits


class SupCEResNet(nn.Module):
    """Encoder + linear classifier (cross-entropy training)."""

    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear (or 2-layer) classifier used by the linear-probe evaluation."""

    def __init__(self, name='resnet50', num_classes=10, two_layers=False):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        if two_layers:
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, num_classes),
            )
        else:
            self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)