import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


"""
NOTE: this can only use cpu because i used python lists which obv was not a good idea in retrospect.
ik you can use a modulelist but i need to store the tensor images. actually i'm not even sure at this point
but i'm just gonna use cpu cuz i've spent too much time and can't figure it out. 
 https://discuss.pytorch.org/t/some-tensors-getting-left-on-cpu-despite-calling-model-to-cuda/112915
"""

class double(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(double,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,3,1,1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels,3,1,1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class cnnModel(nn.Module):

    def __init__(self, input_channels = 3, out_channels = 9):
        super(cnnModel,self).__init__()

        self.pool = nn.MaxPool2d(2,2)
        self.combineList = []
        self.actions = nn.ModuleList()

        self.input_channels = input_channels
        self.output_channels = out_channels

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.doEverything()
        self.sig = nn.Sigmoid()


    def doEverything(self):
        # first convolutional double layer
        func = double(self.input_channels, 64)
        self.actions.append(func)

        func = double(64, 128)
        self.actions.append(func)

        func = double(128, 256)
        self.actions.append(func)


        func = double(256, 512)
        self.actions.append(func)


        func = double(512, 1024)
        self.actions.append(func)
        up = nn.ConvTranspose2d(1024, 512, stride=2, kernel_size=2)
        self.actions.append(up)


        # conv double layer
        func = double(1024, 512)
        self.actions.append(func)

        # convolutional double layer with tranpose
        up = nn.ConvTranspose2d(512, 256, stride=2, kernel_size=2)
        self.actions.append(up)


        # conv double layer
        func = double(512, 256)
        self.actions.append(func)

        # convolutional double layer with tranpose
        up = nn.ConvTranspose2d(256, 128, stride=2, kernel_size=2)
        self.actions.append(up)



        # conv double layer
        func = double(256, 128)
        self.actions.append(func)

        # convolutional double layer with tranpose
        up = nn.ConvTranspose2d(128, 64, stride=2, kernel_size=2)
        self.actions.append(up)

        # conv double layer
        func = double(128, 64)
        self.actions.append(func)


    def forward(self,x):

        ind = 0

        # first convolutional double layer
        func = self.actions[ind]
        ind += 1
        x = func(x)
        self.combineList.append(x)
        x = self.pool(x)

        # second convolutional double layer
        func = self.actions[ind]
        ind += 1
        x = func(x)
        self.combineList.append(x)
        x = self.pool(x)

        # third convolutional double layer
        func = self.actions[ind]
        ind += 1
        x = func(x)

        self.combineList.append(x)
        x = self.pool(x)

        # fourth convolutional double layer
        func = self.actions[ind]
        ind += 1

        x = func(x)

        self.combineList.append(x)
        x = self.pool(x)

        # first convolutional double layer with tranpose
        func = self.actions[ind]
        ind += 1

        x = func(x)

        up = self.actions[ind]
        ind += 1

        x = up(x)

        # combine layer
        add_new = self.combineList[-1]
        self.combineList.pop()
        if x.shape != add_new.shape:
            x = TF.resize(x, size=add_new.shape[2:])

        x = torch.cat((add_new, x), dim=1)

        # conv double layer
        func = self.actions[ind]
        ind += 1

        x = func(x)

        # convolutional double layer with tranpose
        up = self.actions[ind]
        ind += 1

        x = up(x)

        # combine layer
        add_new = self.combineList[-1]
        self.combineList.pop()
        if x.shape != add_new.shape:
            x = TF.resize(x, size=add_new.shape[2:])

        x = torch.cat((add_new, x), dim=1)

        # conv double layer
        func = self.actions[ind]
        ind += 1

        x = func(x)

        # convolutional double layer with tranpose
        up = self.actions[ind]
        ind += 1

        x = up(x)

        # combine layer
        add_new = self.combineList[-1]
        self.combineList.pop()
        if x.shape != add_new.shape:
            x = TF.resize(x, size=add_new.shape[2:])

        x = torch.cat((add_new, x), dim=1)

        # conv double layer
        func = self.actions[ind]
        ind += 1

        x = func(x)

        # convolutional double layer with tranpose
        up = self.actions[ind]
        ind += 1

        x = up(x)

        # combine layer
        add_new = self.combineList[-1]
        self.combineList.pop()
        if x.shape != add_new.shape:
            x = TF.resize(x, size=add_new.shape[2:])

        x = torch.cat((add_new, x), dim=1)

        # conv double layer
        func = self.actions[ind]
        ind += 1

        x = func(x)


        return self.final_conv(x)