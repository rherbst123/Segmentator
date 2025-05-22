import os
import requests
from tqdm import tqdm

def download_huge_checkpoint():
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    os.makedirs("models", exist_ok=True)
    filename = os.path.basename(url)
    local_path = os.path.join("models", filename)

    print(f"Downloading {url} …")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))

    with open(local_path, "wb") as f, tqdm(
        desc=filename,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Saved model checkpoint to {local_path}")

def download_little_checkpoint():
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    os.makedirs("models", exist_ok=True)
    filename = os.path.basename(url)
    local_path = os.path.join("models", filename)

    print(f"Downloading {url} …")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))

    with open(local_path, "wb") as f, tqdm(
        desc=filename,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Saved model checkpoint to {local_path}")


if __name__ == "__main__":
    tprint("Setup")
    print(80*"=")
    print("Welcome to Segmentator setup, lets download the necessary models.")
    print("Would you like to download the vit_h (huge) or the vit_l (little) model?")
    print("Huge model provides better results but does cost more resources to run, little has less precision but faster (better for laptops)")
    
    choice = input("Which model do you want to download? (huge / little): ")

    if choice == "huge":
        download_huge_checkpoint()
    elif choice == "little":
        download_little_checkpoint()
