import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

IMG_SIZE = 256
BATCH_SIZE = 4
DATA_DIR = "data"
MODEL_PATH = "autoencoder_fresh_veggies.pth"
DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=transform
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

criterion = nn.MSELoss(reduction="none")

errors = []

with torch.no_grad():
    for imgs, _ in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        loss = loss.mean(dim=[1, 2, 3]) 
        errors.extend(loss.cpu().numpy())

errors = np.array(errors)

threshold = errors.mean() + 2 * errors.std()

print("Validation Mean Error:", errors.mean())
print("Validation Std Error:", errors.std())
print("Anomaly Threshold:", threshold)

np.save("threshold.npy", threshold)
