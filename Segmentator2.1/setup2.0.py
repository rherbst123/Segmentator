#!/usr/bin/env python3
"""
Segmentator — SAM 2.1 model downloader

• “small”  (≈184 MB) → sam2.1_hiera_small.pt  +  sam2.1_hiera_s.yaml  
• “large”  (≈898 MB) → sam2.1_hiera_large.pt  +  sam2.1_hiera_l.yaml

The files are saved next to this script in  <repo-root>/models/.
"""

import os
import requests
from tqdm import tqdm
from art import tprint

# ------------------------------------------------------------------
# URLs for SAM 2.1 weights + configs
# ------------------------------------------------------------------
MODELS = {
    "small": {
        "ckpt_url":  (
            "https://huggingface.co/facebook/sam2.1-hiera-small/"
            "resolve/main/sam2.1_hiera_small.pt"
        ),
        "yaml_url": (
            "https://raw.githubusercontent.com/facebookresearch/"
            "sam2/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
        ),
        "ckpt_name": "sam2.1_hiera_small.pt",
        "yaml_name": "sam2.1_hiera_s.yaml",
    },
    "large": {
        "ckpt_url": (
            "https://huggingface.co/facebook/sam2.1-hiera-large/"
            "resolve/main/sam2.1_hiera_large.pt"
        ),
        "yaml_url": (
            "https://raw.githubusercontent.com/facebookresearch/"
            "sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        ),
        "ckpt_name": "sam2.1_hiera_large.pt",
        "yaml_name": "sam2.1_hiera_l.yaml",
    },
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def dl_file(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"⬇️  Downloading {os.path.basename(dst)}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(dst),
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))


def grab(model_key: str) -> None:
    entry = MODELS[model_key]
    root = os.path.join(os.path.dirname(__file__), "models")
    ckpt_path = os.path.join(root, entry["ckpt_name"])
    yaml_path = os.path.join(root, entry["yaml_name"])

    # checkpoint
    if not os.path.exists(ckpt_path):
        dl_file(entry["ckpt_url"], ckpt_path)
    else:
        print(f"✅  {entry['ckpt_name']} already exists")

    # YAML
    if not os.path.exists(yaml_path):
        dl_file(entry["yaml_url"], yaml_path)
    else:
        print(f"✅  {entry['yaml_name']} already exists")

    print("\n🎉  Done!  Files saved to:", root)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    tprint("Setup")
    print("=" * 80)
    print("Welcome to Segmentator SAM 2.1 setup.")
    print("Models:")
    print("  • small  – fast, ~184 MB VRAM-friendly")
    print("  • large  – highest accuracy, ~898 MB")
    print()

    while True:
        choice = input("Which model do you want to download? (small / large): ").strip().lower()
        if choice in MODELS:
            grab(choice)
            break
        else:
            print("Please type 'small' or 'large'.")
