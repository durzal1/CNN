import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import *
from dataset import *
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from model2 import UNET
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "final.pth.tar"
IMG_DIR = "football/images"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 200


def save_checkpoint(state, filename="final.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()

def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # val, _ = torch.max(preds, dim=1)
            softmax = nn.Softmax(dim=1 )
            preds = torch.argmax(softmax(model(x)), axis=1).to(device)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device=DEVICE
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        fig, ax = plt.subplots(3, figsize=(18, 18))

        preds = model(x)
        val, _ = torch.max(preds, dim=1)
        softmax = nn.Softmax(dim=1)
        preds = torch.argmax(softmax(model(x)), axis=1).to('cpu')
        img1 = np.transpose(np.array(x[0, :, :, :].to('cpu')), (1, 2, 0))
        preds1 = np.array(preds[0, :, :])
        mask1 = np.array(y[0, :, :])
        ax[0].imshow(img1)
        ax[1].imshow(preds1)
        ax[2].imshow(mask1)
        plt.savefig(f"{folder}real{idx}.png")
        # plt.show()

    model.train()

transform = A.Compose(
    [
        A.Resize(width=IMAGE_HEIGHT, height=IMAGE_WIDTH),
        # A.RandomCrop(width=1280, height=720),
        # A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
        # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9), # don't think i need this tbh
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


# DEVICE = 'cpu'
model = cnnModel().to(DEVICE)
model.cuda()

# model = UNET(in_channels=3, out_channels=9).to(DEVICE)

optimizer = optim.Adam(
     model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
loss_fn = nn.CrossEntropyLoss()

load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)


train_dataset = footballDataset(
        transform=transform,
        img_dir=IMG_DIR,
        names= np.arange(start=1, stop=100),
)

# test_dataset = VOCDataset(
#         "yolo_images/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
# )

train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
)

# test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY,
#         shuffle=True,
#         drop_last=True,
#  )


save_predictions_as_imgs(
         train_loader, model, folder="saved_images/", device=DEVICE
    )

for epoch in range(EPOCHS):

    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)


        out = model(x)

        loss = loss_fn(out, y)


        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # update progress bar
        loop.set_postfix(loss=loss.item())

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


    check_accuracy(train_loader, model, device=DEVICE)

    # # # print some examples to a folder
    # save_predictions_as_imgs(
    #      train_loader, model, folder="saved_images/", device=DEVICE
    # )



    if (epoch % 1 == 0):
        print("[INFO] EPOCH: {}/{}".format(epoch + 1, EPOCHS))
        print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


save_checkpoint(checkpoint)
