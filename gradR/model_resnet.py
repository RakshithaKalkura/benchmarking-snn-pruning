# model_resnet.py
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron

class ResNetBasicBlockSNN(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, v_threshold=1.0, v_reset=0.0, tau=2.0, surrogate_function=surrogate.ATan()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau,
                                   surrogate_function=surrogate_function, detach_reset=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau,
                                   surrogate_function=surrogate_function, detach_reset=True)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        # x expected to be spiking input (per timestep) or static conv followed by lif
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.lif2(out + identity)
        return out

class ResNet19Net(nn.Module):
    """
    ResNet19-like SNN for CIFAR10 producing aggregated spike-count logits like Cifar10Net.
    Forward returns: out_spikes_counter (Tensor) shape [N, num_classes]
    """
    def __init__(self, T=8, num_classes=10, v_threshold=1.0, v_reset=0.0, tau=2.0, surrogate_function=surrogate.ATan()):
        super().__init__()
        self.T = int(T)
        self.train_times = 0
        self.epochs = 0
        self.max_test_acccuracy = 0

        # initial conv
        self.static_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        # layers - construct something like [2,2,2,2] layout but adjust to get 8x8 at end
        self.in_planes = 64
        self.layer1 = self._make_layer(ResNetBasicBlockSNN, 64, 2, stride=1, v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function)
        self.layer2 = self._make_layer(ResNetBasicBlockSNN, 128, 2, stride=2, v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function)
        self.layer3 = self._make_layer(ResNetBasicBlockSNN, 256, 2, stride=2, v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function)
        self.layer4 = self._make_layer(ResNetBasicBlockSNN, 256, 2, stride=2, v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function)

        # final classifier implemented similarly to Cifar10Net: map to num_classes * 10 then boost
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # will reduce spatial dims to 1x1
        fc_in = 256 * ResNetBasicBlockSNN.expansion  # channels
        # map to an internal fc-sized vector (keep simple)
        self.fc1 = nn.Linear(fc_in, 128 * 4 * 4, bias=False)
        self.lif_fc = neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau,
                                     surrogate_function=surrogate_function, detach_reset=True)
        self.fc2 = nn.Linear(128 * 4 * 4, num_classes * 10, bias=False)
        self.lif_out = neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau,
                                      surrogate_function=surrogate_function, detach_reset=True)
        self.boost = nn.AvgPool1d(10, 10)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride, v_threshold, v_reset, tau, surrogate_function):
        layers = []
        layers.append(block(self.in_planes, planes, stride=stride, v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: [N, C, H, W]
        returns: out_spikes_counter [N, num_classes] (float counts / rates)
        """
        # static conv done once (same as your Cifar10Net)
        static_x = self.static_conv(x)  # [N, C, H, W] static features (BN applied)
        # We'll run the spiking pipeline for T timesteps, accumulate outputs
        out_spikes_counter = None
        for t in range(self.T):
            # run residual layers (each block contains lif nodes)
            out = static_x
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            out = self.avgpool(out)  # [N, C, 1, 1]
            out = out.view(out.size(0), -1)  # [N, C]
            out = self.lif_fc(self.fc1(out))
            out = self.lif_out(self.fc2(out))  # [N, num_classes*10]

            # average pool across groups of 10 -> boost to get num_classes (matching Cifar10Net)
            logits_t = self.boost(out.unsqueeze(1)).squeeze(1)  # [N, num_classes]
            if out_spikes_counter is None:
                out_spikes_counter = logits_t
            else:
                out_spikes_counter = out_spikes_counter + logits_t

        return out_spikes_counter  # sums across T timesteps; downstream expects this format
