import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import torchvision.transforms as T

# This transform mirrors load_CelebA:
# ToTensor -> Resize(img_size) -> CenterCrop(img_size) -> Grayscale(1)
IMG_SIZE = 32

transform = T.Compose([
    T.ToTensor(),
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.Grayscale(num_output_channels=1),
])

# Directory where your existing Google Drive download put the files
# (img_align_celeba, list_attr_celeba.txt, etc.).
root = "celeba"

# We need some target_type that CelebA accepts; we ignore targets.
ds = CelebA(
    root=root,
    split="train",
    target_type="attr",   # required; will be ignored
    transform=transform,
    download=False,       # you already downloaded with this script
)

loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)

imgs_list = []
for imgs, _ in loader:        # imgs: (B, 1, 32, 32)
    imgs_list.append(imgs)

all_imgs = torch.cat(imgs_list, dim=0)   # (N, 1, 32, 32)

# Save in the path cfg.py expects: config.path_data + 'CelebA{size}.pt'
# With config.DATASET='CelebA' and size=32, that is ../CelebA32.pt from raw/
torch.save(all_imgs, "../CelebA32.pt")
print("Saved", all_imgs.shape, "to ../CelebA32.pt")
