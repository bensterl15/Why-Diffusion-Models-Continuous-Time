import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import torchvision.transforms as T
from torchvision.utils import save_image

IMG_SIZE = 32

transform = T.Compose([
    T.ToTensor(),
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.Grayscale(num_output_channels=1),
])

root = "celeba"  # same as your download.py
out_dir = "fid_celeba32_imgs"  # folder for FID stats input
os.makedirs(out_dir, exist_ok=True)

ds = CelebA(
    root=root,
    split="train",
    target_type="attr",
    transform=transform,
    download=False,
)

loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)

idx = 0
for imgs, _ in loader:          # imgs: (B, 1, 32, 32)
    for img in imgs:
        save_image(img, os.path.join(out_dir, f"{idx:06d}.png"))
        idx += 1

print("Saved", idx, "images to", out_dir)
