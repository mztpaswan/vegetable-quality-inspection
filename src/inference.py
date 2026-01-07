import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

IMG_SIZE = 256
TEST_DIR = "data/test/mixed"
MODEL_PATH = "autoencoder_fresh_veggies.pth"
THRESHOLD_PATH = "threshold.npy"
DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

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

threshold = np.load(THRESHOLD_PATH)

criterion = nn.MSELoss(reduction="mean")

print("\nInference Results:\n")

for img_name in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img_name)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon = model(img_tensor)
        error = criterion(recon, img_tensor).item()

    label = "FRESH" if error <= threshold else "NOT USABLE"

    print(f"{img_name:30s} | Error: {error:.6f} | {label}")
