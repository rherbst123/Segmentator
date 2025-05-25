# Segmentator

Segmentator is a tool for automated image segmentation and preparation of transcription-ready collages, designed for use with the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything). It downloads images from a list of URLs, enhances them, segments them using SAM, and organizes the results into collages and folders for easy transcription.

## Features

- Download images from `.txt` or `.csv` lists of URLs
- Enhance images for optimal segmentation
- Segment images using Meta's Segment Anything Model (SAM)
- Filter and organize segmented images
- Create collages and transcription-ready folders

## Requirements

- Python 3.8+
- NVIDIA GPU with at least 8GB VRAM (recommended for best performance)
- CUDA drivers (if using GPU)

### Python Dependencies

- torch
- torchvision
- opencv-python
- numpy
- tqdm
- pillow
- requests
- art
- easyocr
- psutil
- GPUtil
- segment-anything (install from [SAM GitHub](https://github.com/facebookresearch/segment-anything))

Install dependencies with:

```sh
pip install torch torchvision opencv-python numpy tqdm pillow requests art easyocr psutil GPUtil
# For SAM, follow instructions at https://github.com/facebookresearch/segment-anything
```

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/Segmentator.git
    cd Segmentator
    ```

2. **Download the SAM model checkpoint:**

    Run the setup script to download the required model weights:

    ```sh
    python setup.py
    ```

    This will download the `sam_vit_h_4b8939.pth` checkpoint into `Segmentator/models/`.

3. **Prepare your image URL list:**

    - Create a `.txt` or `.csv` file containing image URLs (one per line for `.txt`, or comma-separated for `.csv`).

4. **Run Segmentator:**

    ```sh
    python Segmentator/segmentator.py
    ```

    Follow the prompts to:
    - Provide your URL list file
    - Optionally view and enhance images
    - Run segmentation and generate collages

5. **Results:**

    - Enhanced images: `Segmentator/enhanced-images/`
    - Segmented images: `Segmentator/segmented-images/`
    - Transcription-ready collages and folders: `~/Desktop/Transcription_Ready_Images/<timestamp>/`

## Notes

- The first run may take time to download models and process images.
- GPU is highly recommended for segmentation speed.
- For best results, use high-quality input images.



**Developed by Riley Herbst**