# Segmentator

A tool for segmenting images using Meta's Segment Anything Model (SAM).

## System Requirements

- Python 3.8+
- PyTorch 1.7+
- CUDA-compatible GPU with at least 4GB VRAM (recommended)
- 8GB+ RAM

## Installation

1. Clone this repository:
```bash
git clone https://github.com/FieldMuseum/Segmentator.git
cd Segmentator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the SAM models:
```bash
python download_models.py
```

## Usage

1. Prepare a text file with image URLs (one URL per line) or a CSV file with URLs.

2. Run the segmentator:
```bash
python -m Segmentator.segmentator
```

3. Follow the prompts to:
   - Provide the path to your URL file
   - View downloaded images
   - Enhance images
   - Choose a SAM model
   - Run segmentation

## Memory Optimization

If you're experiencing memory issues:

1. Use the ViT-B model (option 3) which requires the least memory
2. Reduce image resolution by modifying the `enhance_image` function
3. Process fewer images at a time

## Troubleshooting

- **Out of memory errors**: The script will now automatically adjust settings based on your hardware. If you still encounter memory issues, try using the ViT-B model.
- **CUDA errors**: If you encounter CUDA errors, the script will automatically fall back to CPU processing.

## License

[MIT License](LICENSE)