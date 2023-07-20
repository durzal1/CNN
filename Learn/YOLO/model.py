"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
Based off of https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    (1, 256, 1, 0), (3, 512, 1, 1),
    (1, 256, 1, 0), (3, 512, 1, 1),
    (1, 256, 1, 0), (3, 512, 1, 1),
    (1, 256, 1, 0), (3, 512, 1, 1),
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    (1, 512, 1, 0), (3, 1024, 1, 1),
    (1, 512, 1, 0), (3, 1024, 1, 1),
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False,**kwargs)
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class cnnArchitecture(nn.Module):

    def __init__(self, input_channels = 3, **kwargs):
        super(cnnArchitecture, self).__init__()
        self.architecture = architecture_config
        self.input_channels = input_channels
        self.darknet = self._create_conv_layers()

        self.fcs = self.createFC(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self):
        layers = []

        input_channels = self.input_channels

        for x in self.architecture:
            if type(x) == tuple:
                ## (kernel_size, filters, stride, padding)
                layers += [
                    CNNBlock(input_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])
                ]
                input_channels = x[1]

            elif x == 'M':
                layers += [
                    nn.MaxPool2d(2, 2)
                ]

        func = nn.Sequential(*layers)
        return func

    def createFC(self, split_size, num_boxes, num_classes):
        # FC layers

        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),  # idk why we need this
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )



