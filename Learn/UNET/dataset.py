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


        """
        Red = 0, Purple = 1, Yellow = 2, Orange = 3, Blue = 4, Gray= 5, Pink = 6, Peach = 7
        """


        augmentations = self.transform(image=img, mask=result)
        augmented_img = augmentations["image"]
        augmented_result = augmentations["mask"]

        # Create dummy target image
        h, w = 160, 200
        result = augmented_result


        colors = np.array([[0,0,0],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]], np.int32)

        colors[0] = (219, 12, 85)
        colors[1] = (255, 0, 29)
        colors[2] = (111, 48, 253)
        colors[3] = (254, 233, 3)
        colors[4] = (255, 160, 1)
        colors[5] = (27, 71, 151)
        colors[6] = (137, 126, 126)
        colors[7] = (201, 19, 223)
        colors[8] = (238, 171, 171)
        result = result.permute(2, 1, 0).contiguous()

        mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

        mask = torch.empty(h, w, dtype=torch.long)
        mask[mask<0] = 0
        mask[mask>8] = 0
        for k in mapping:
            # Get all indices for current class
            idx = (result == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)  # Check that all channels match
            mask[validx] = torch.tensor(mapping[k], dtype=torch.long)

        mask = mask.permute(1,0).contiguous()
        # visualize(augmented_img)
        # visualize(augmented_result)
        # print(mask)




        return augmented_img, mask