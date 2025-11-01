#vgg16 models

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer
from .submodules.sparse import ConvBlock
from .submodules.layers import conv1x1, conv3x3, create_mslif

__all__ = ['VGGBlockSNN', 'VGG16SNN']

class VGGBlockSNN(nn.Module):
    """
    A VGG conv block with n conv layers (each conv3x3 + BN + spiking neuron block).
    """
    def __init__(self, in_channels, out_channels, num_convs, T=8, sparse_weights=True, sparse_neurons=True):
        super().__init__()
        layers = []
        for i in range(num_convs):
            inc = in_channels if i == 0 else out_channels
            layers.append(
                ConvBlock(
                    conv3x3(inc, out_channels),
                    nn.BatchNorm2d(out_channels),
                    create_mslif(),
                    sparse_weights=sparse_weights,
                    sparse_neurons=sparse_neurons,
                    T=T
                )
            )
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)

    def connects(self, sparse, dense):
        conn, total = 0, 0
        for layer in self.convs:
            c, t, sparse, dense = layer.connects(sparse, dense)
            conn, total = conn + c, total + t
        return conn, total, sparse, dense

    def calc_c(self, x):
        for layer in self.convs:
            x = layer.calc_c(x)
        return x


class VGG16SNN(nn.Module):
    """
    VGG16-like SNN for CIFAR-10 using ConvBlock wrappers.
    Produces per-timestep logits [T, N, num_classes] same as other models.
    """
    def __init__(self, T=8, base_width=64, num_classes=10):
        super().__init__()
        self.T = T
        self.skip = []  # keep consistent with other nets

        # VGG config: 2x64, 2x128, 3x256, 3x512, 3x512
        self.block1 = VGGBlockSNN(3, base_width, num_convs=2, T=T,
                                   sparse_weights=True, sparse_neurons=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32->16

        self.block2 = VGGBlockSNN(base_width, base_width*2, num_convs=2, T=T,
                                   sparse_weights=True, sparse_neurons=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16->8

        self.block3 = VGGBlockSNN(base_width*2, base_width*4, num_convs=3, T=T,
                                   sparse_weights=True, sparse_neurons=True)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8->4

        self.block4 = VGGBlockSNN(base_width*4, base_width*8, num_convs=3, T=T,
                                   sparse_weights=True, sparse_neurons=True)
        self.pool4 = nn.MaxPool2d(2, 2)  # 4->2

        self.block5 = VGGBlockSNN(base_width*8, base_width*8, num_convs=3, T=T,
                                   sparse_weights=True, sparse_neurons=True)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))  # global pool to 1x1

        # Following your project's pattern: convert pooled features -> conv1x1 -> classifier conv
        fc_in = base_width * 8
        # optional intermediate projection (kept simple)
        self.fc_conv = ConvBlock(
            conv1x1(fc_in, fc_in),
            None,
            create_mslif(),
            sparse_weights=True,
            sparse_neurons=False,
            T=T
        )

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

            c, t, sparse, dense = self.block1.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool1(sparse), self.pool1(dense)

            c, t, sparse, dense = self.block2.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool2(sparse), self.pool2(dense)

            c, t, sparse, dense = self.block3.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool3(sparse), self.pool3(dense)

            c, t, sparse, dense = self.block4.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool4(sparse), self.pool4(dense)

            c, t, sparse, dense = self.block5.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.pool5(sparse), self.pool5(dense)

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
            x = self.block1.calc_c(x)
            x = self.pool1(x)
            x = self.block2.calc_c(x)
            x = self.pool2(x)
            x = self.block3.calc_c(x)
            x = self.pool3(x)
            x = self.block4.calc_c(x)
            x = self.pool4(x)
            x = self.block5.calc_c(x)
            x = self.pool5(x)
            x = x.view(1, -1, 1, 1)
            x = self.fc_conv.calc_c(x)
            x = self.classifier.calc_c(x)

    def forward(self, x):
        # Input x: [N, C, H, W] -> output: [T, N, num_classes]
        x = self.block1(x)
        x = functional.seq_to_ann_forward(x, self.pool1)
        x = self.block2(x)
        x = functional.seq_to_ann_forward(x, self.pool2)
        x = self.block3(x)
        x = functional.seq_to_ann_forward(x, self.pool3)
        x = self.block4(x)
        x = functional.seq_to_ann_forward(x, self.pool4)
        x = self.block5(x)
        x = functional.seq_to_ann_forward(x, self.pool5)  # [T, N, C, 1, 1]
        x = x.view(x.shape[0], x.shape[1], -1, 1, 1)
        x = self.fc_conv(x)
        x = self.classifier(x)
        x = x.flatten(2)  # [T, N, L]
        x = x.unsqueeze(2)  # [T, N, 1, L]
        out = functional.seq_to_ann_forward(x, self.boost).squeeze(2)  # [T, N, num_classes]
        return out
