import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms




# Hyper-parameters
num_epochs = 5
batch_size = 64
learning_rate = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset_trian = datasets.MNIST('../data', train=True,
                          transform=transform)
dataset_test = datasets.MNIST('../data', train=False,
                          transform=transform)

loader_train = torch.utils.data.DataLoader(dataset_trian, batch_size=batch_size,
                                           shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                           shuffle=True)
def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(loader_train)
images, labels = next(dataiter)

# show images
# imshow(torchvision.utils.make_grid(images))

# Network varaibles
# 28x28x1 pixel size -> 24x24x6 -> 20x20x20
# ok so this is ignoring pooling layers so it's actually not true
conv1_output_channels = 20
conv2_output_channels = 40


class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(1,conv1_output_channels, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(conv1_output_channels,conv2_output_channels,5)
        self.fc1 = nn.Linear(4*4 * conv2_output_channels,120)
        self.fc2 = nn.Linear(120,10)

    def forward(self,x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,conv2_output_channels*4*4) # 40 channels. 4x4 dimension i got from debugger mode
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(x,dim=1)
        return output

model = convNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_total_steps = len(loader_train)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(loader_train):

        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')


print('Finished Training')


# Testing

with torch.no_grad():
    correct = 0
    samples = 0

    for images, labels in loader_test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)

        test = labels.size(0)
        samples += labels.size(0) # how many samples in it
        correct += (predicted == labels).sum().item()


    acc = 100.0 * correct / samples
    print(f'Accuracy of the network: {acc} %')