import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class double(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(double,self).__init__()
        self.func = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,3, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels,3, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.func(x)

class cnnModel(nn.Module):

    def __init__(self, input_channels = 1, out_channels = 1):
        super(cnnModel,self).__init__()

        self.pool = nn.MaxPool2d(2,2)
        self.combineList = []

        self.input_channels = input_channels
        self.output_channels = out_channels

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)




    def forward(self,x):


        # first convolutional double layer
        func = double(1,64)
        x = func(x)
        self.combineList.append(x)
        x = self.pool(x)

        # second convolutional double layer
        func = double(64, 128)
        x = func(x)
        self.combineList.append(x)
        x = self.pool(x)

        # third convolutional double layer
        func = double(128, 256)
        x = func(x)
        self.combineList.append(x)
        x = self.pool(x)

        # fourth convolutional double layer
        func = double(256, 512)
        x = func(x)
        self.combineList.append(x)
        x = self.pool(x)

        # first convolutional double layer with tranpose
        func = double(512, 1024)
        x = func(x)
        up = nn.ConvTranspose2d(1024,512, stride=2,kernel_size=2)
        x = up(x)

        # combine layer
        add_new = self.combineList[-1]
        self.combineList.pop()
        if x.shape != add_new.shape:
            x = TF.resize(x, size=add_new.shape[2:])

        x = torch.cat((add_new, x), dim=1)

        # conv double layer
        func = double(1024, 512)
        x = func(x)

        # convolutional double layer with tranpose
        up = nn.ConvTranspose2d(512, 256, stride=2, kernel_size=2)
        x = up(x)

        # combine layer
        add_new = self.combineList[-1]
        self.combineList.pop()
        if x.shape != add_new.shape:
            x = TF.resize(x, size=add_new.shape[2:])

        x = torch.cat((add_new, x), dim=1)

        # conv double layer
        func = double(512, 256)
        x = func(x)

        # convolutional double layer with tranpose
        up = nn.ConvTranspose2d(256, 128, stride=2, kernel_size=2)
        x = up(x)

        # combine layer
        add_new = self.combineList[-1]
        self.combineList.pop()
        if x.shape != add_new.shape:
            x = TF.resize(x, size=add_new.shape[2:])

        x = torch.cat((add_new, x), dim=1)

        # conv double layer
        func = double(256, 128)
        x = func(x)

        # convolutional double layer with tranpose
        up = nn.ConvTranspose2d(128, 64, stride=2, kernel_size=2)
        x = up(x)

        # combine layer
        add_new = self.combineList[-1]
        self.combineList.pop()
        if x.shape != add_new.shape:
            x = TF.resize(x, size=add_new.shape[2:])

        x = torch.cat((add_new, x), dim=1)

        # conv double layer
        func = double(128, 64)
        x = func(x)

        return self.final_conv(x)


x = torch.randn((3, 1, 160, 160))
model = cnnModel()
preds = model(x)

assert preds.shape == x.shape