import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from .submodules.sparse import ConvBlock
from .submodules.layers import conv1x1, conv3x3, create_mslif

__all__ = ['ResNetBasicBlockSNN', 'ResNet19SNN']

class ResNetBasicBlockSNN(nn.Module):
    """
    Basic residual block for ResNet-style SNN using ConvBlock wrappers.
    If downsample is True, performs spatial downsample with stride=2 conv.
    """
    expansion = 1

    def __init__(self, in_planes, planes, T=8, downsample=False, sparse_weights=True, sparse_neurons=True):
        super().__init__()
        stride = 2 if downsample else 1

        # first conv in block
        self.conv1 = ConvBlock(
            conv3x3(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            create_mslif(),
            sparse_weights=sparse_weights,
            sparse_neurons=sparse_neurons,
            T=T
        )
        # second conv in block
        self.conv2 = ConvBlock(
            conv3x3(planes, planes, 1),
            nn.BatchNorm2d(planes),
            create_mslif(),
            sparse_weights=sparse_weights,
            sparse_neurons=sparse_neurons,
            T=T
        )

        # downsample path if needed
        self.downsample = None
        if downsample or in_planes != planes * self.expansion:
            # a simple conv1x1 to match channels and spatial size
            self.downsample = ConvBlock(
                conv1x1(in_planes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion),
                create_mslif(),
                sparse_weights=sparse_weights,
                sparse_neurons=False,  # usually residual bypass is not spiking
                T=T
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        # elementwise add (spiking tensors are sequences: [T,N,C,H,W])
        # ensure shapes match
        out = out + identity
        return out

    def connects(self, sparse, dense):
        conn, total = 0, 0
        c, t, sparse, dense = self.conv1.connects(sparse, dense)
        conn, total = conn + c, total + t
        c, t, sparse, dense = self.conv2.connects(sparse, dense)
        conn, total = conn + c, total + t
        if self.downsample is not None:
            c, t, sparse, dense = self.downsample.connects(sparse, dense)
            conn, total = conn + c, total + t
        return conn, total, sparse, dense

    def calc_c(self, x, prev_layers=None):
        x = self.conv1.calc_c(x, prev_layers or [])
        x = self.conv2.calc_c(x, [self.conv1])
        if self.downsample is not None:
            x = self.downsample.calc_c(x)
        return x


class ResNet19SNN(nn.Module):
    """
    ResNet19-like SNN for CIFAR-10 using ConvBlock wrappers.
    Layer naming follows the order of module registration (useful for profiler).
    """
    def __init__(self, T=8, base_width=64, num_classes=10):
        super().__init__()
        self.T = T
        self.in_planes = base_width

        # initial static conv (no temporal dynamics for input layer)
        self.static_conv = ConvBlock(
            conv3x3(3, base_width),
            nn.BatchNorm2d(base_width),
            create_mslif(),
            static=True, T=T,
            sparse_weights=True, sparse_neurons=False
        )

        # residual layers: each layer contains N basic blocks
        # using block counts [2,2,2,2] as ResNet-19 (similar to ResNet-18 structure)
        self.layer1 = self._make_layer(ResNetBasicBlockSNN, base_width, 2, stride=1)
        self.layer2 = self._make_layer(ResNetBasicBlockSNN, base_width*2, 2, stride=2)
        self.layer3 = self._make_layer(ResNetBasicBlockSNN, base_width*4, 2, stride=2)
        self.layer4 = self._make_layer(ResNetBasicBlockSNN, base_width*8, 2, stride=2)

        # global pooling and fc implemented with ConvBlock -> boost pool
        # flatten spatial dims into channel dim before fc conv1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        fc_in = base_width * 8 * ResNetBasicBlockSNN.expansion  # channels after layer4
        # convert channel vector to time-distributed linear via conv1x1 wrapped in ConvBlock
        # We follow your project's pattern: make fc as ConvBlock producing num_classes*10 then AvgPool1d(10)
        self.fc_conv = ConvBlock(
            conv1x1(fc_in, (fc_in // 2) * 4 * 4) if False else conv1x1(fc_in, fc_in),  # keep it simple
            None,
            create_mslif(),
            sparse_weights=True,
            sparse_neurons=False,
            T=T
        )
        # final classifier (map to num_classes*10 for boosting)
        self.classifier = ConvBlock(
            conv1x1(fc_in, num_classes * 10),
            None,
            create_mslif(),
            sparse_weights=True,
            sparse_neurons=False,
            T=T
        )
        self.boost = nn.AvgPool1d(10, 10)

        self.init_weight()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        # first block may downsample
        layers.append(block(self.in_planes, planes, T=self.T, downsample=(stride!=1)))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, T=self.T))
        return nn.Sequential(*layers)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def connects(self):
        conn, total = 0, 0
        with torch.no_grad():
            sparse = torch.ones(1, 3, 32, 32, device='cuda')
            dense = torch.ones(1, 3, 32, 32, device='cuda')

            c, t, sparse, dense = self.static_conv.connects(sparse, dense)
            conn, total = conn + c, total + t

            # layer1
            for block in self.layer1:
                c, t, sparse, dense = block.connects(sparse, dense)
                conn, total = conn + c, total + t

            # layer2
            for block in self.layer2:
                c, t, sparse, dense = block.connects(sparse, dense)
                conn, total = conn + c, total + t

            # layer3
            for block in self.layer3:
                c, t, sparse, dense = block.connects(sparse, dense)
                conn, total = conn + c, total + t

            # layer4
            for block in self.layer4:
                c, t, sparse, dense = block.connects(sparse, dense)
                conn, total = conn + c, total + t

            # final fc convs
            # reshape: average pool then view to [1, C, 1, 1] as in your earlier nets
            sparse = self.avgpool(sparse)
            dense = self.avgpool(dense)
            sparse = sparse.view(1, -1, 1, 1)
            dense = dense.view(1, -1, 1, 1)
            c, t, sparse, dense = self.fc_conv.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.classifier.connects(sparse, dense)
            conn, total = conn + c, total + t

        return conn, total

    def calc_c(self):
        with torch.no_grad():
            x = torch.ones(1, 3, 32, 32, device='cuda')
            x = self.static_conv.calc_c(x)
            for block in self.layer1:
                x = block.calc_c(x)
            for block in self.layer2:
                x = block.calc_c(x)
            for block in self.layer3:
                x = block.calc_c(x)
            for block in self.layer4:
                x = block.calc_c(x)
            x = self.avgpool(x)
            x = x.view(1, -1, 1, 1)
            x = self.fc_conv.calc_c(x)
            x = self.classifier.calc_c(x)

    def forward(self, x):
        # following your Cifar10Net forward style: inputs are [N, C, H, W] -> convs produce [T, N, C, H, W]
        x = self.static_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # global pooling; collapse spatial dims into feature vector
        # depending on ConvBlock outputs shape, use seq_to_ann_forward to apply avgpool per timestep
        x = functional.seq_to_ann_forward(x, self.avgpool)  # [T, N, C, 1, 1]
        x = x.view(x.shape[0], x.shape[1], -1, 1, 1)  # [T, N, Cx1x1,1,1] keep interface similar
        x = self.fc_conv(x)
        x = self.classifier(x)
        x = x.flatten(2)  # [T, N, L]
        x = x.unsqueeze(2)  # [T, N, 1, L]
        out = functional.seq_to_ann_forward(x, self.boost).squeeze(2)  # [T, N, num_classes]
        return out
