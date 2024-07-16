import os
import torch.nn as nn
import torch.nn.functional as F
import torch

from imitation_shared.utils import *


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        self.img_fc = nn.Sequential(
            nn.Linear(6400, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        self.scalar_fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.emb_fc = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Tanh()
        )

    def forward(self, x_img, scalar):
        x_img = self.conv_blocks(x_img)
        x_img = x_img.view(x_img.size(0), -1)
        x_img = self.img_fc(x_img)

        scalar = self.scalar_fc(scalar)

        emb = torch.cat([x_img, scalar], dim=1)
        emb = self.emb_fc(emb)

        output = self.fc(emb)

        return output


def save_model(folder, model, name):
    """
    Saves the model to the specified folder with the specified name.

    Parameters:
        folder (str): The folder in which to save the model.
        model (torch.nn.Module): The model to save.
        name (str): The name to use for the saved model file.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    torch.save(model.state_dict(), os.path.join(folder, f"{name}.pth"))
    print_formatted(f"Model saved to {folder}/{name}.pth", GREEN)


def load_model(folder, name):
    """
    Loads the model from the specified folder with the specified name.

    Parameters:
        folder (str): The folder from which to load the model.
        name (str): The name of the model file to load.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = CNNModel()

    try:
        model.load_state_dict(torch.load(os.path.join(folder, f"{name}.pth")))
        print_formatted(f"Model loaded from {folder}/{name}.pth", GREEN)
    except FileNotFoundError:
        print_formatted(f"Existing model not found", RED)

    return model
