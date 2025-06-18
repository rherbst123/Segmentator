#!/usr/bin/env python3
import os
import requests
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file from a URL to a destination with progress bar
    """
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    tprint("Setup")
    print(85*"=")
    print("This script will download the models for the Segmentator, give it a second")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), "Segmentator", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Model URLs
    model_urls = {
        "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    # Download models
    for model_name, url in model_urls.items():
        destination = os.path.join(models_dir, model_name)
        download_file(url, destination)
    
    print("\nAll models downloaded successfully!")
    print("\nModel sizes and recommended hardware:")
    print("- ViT-B: ~375MB, recommended for systems with limited resources")
    print("- ViT-L: ~1.2GB, recommended for systems with moderate resources")
    print("- ViT-H: ~2.5GB, recommended for systems with high-end resources")

if __name__ == "__main__":
    main()