import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()

class footballDataset(torch.utils.data.Dataset):
    def __init__(
            self, names, img_dir, transform
    ):
        self.names = names
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        i = idx
        name = self.names[i]

        img_path = f"{self.img_dir}/Frame 1  ({name}).jpg"
        result_path = f"{self.img_dir}/Frame 1  ({name}).jpg___fuse.png"
        img = np.array(Image.open(img_path).convert("RGB"))
        result = np.array(Image.open(result_path).convert("RGB"))


        augmentations = self.transform(image=img)
        augmented_img = augmentations["image"]

        augmentations = self.transform(image=result)
        augmented_result = augmentations["image"]

        # visualize(augmented_img)
        # visualize(augmented_result)

        return augmented_img, augmented_result