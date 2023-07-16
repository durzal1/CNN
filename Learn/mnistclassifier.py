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
learning_rate = 0.01

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
imshow(torchvision.utils.make_grid(images))


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(loader_train):

        images = images.to(device)
        labels = labels.to(device)

        print(images)
        print('f')
