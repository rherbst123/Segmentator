import os
import shutil
import requests
import csv
from art import *
from tqdm import tqdm
from PIL import Image, ImageEnhance
import psutil
import torch
import cv2
import numpy as np
gc = __import__('gc')
import GPUtil
import threading
import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.ops import boxes
import easyocr
from datetime import datetime

# Patch for boxes.batched_nms
_original_batched_nms = boxes.batched_nms

def patched_batched_nms(boxes_tensor, scores, idxs, iou_threshold):
    boxes_tensor = boxes_tensor.cpu()
    scores = scores.cpu()
    idxs = idxs.cpu()
    return _original_batched_nms(boxes_tensor, scores, idxs, iou_threshold)

boxes.batched_nms = patched_batched_nms

# Utility functions for resource monitoring
def get_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_percent = gpu.load * 100
        gpu_memory_used = gpu.memoryUsed
        gpu_memory_total = gpu.memoryTotal
        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
    else:
        gpu_percent = 0
        gpu_memory_percent = 0
    return f"CPU:{cpu_percent:.1f}%, Mem:{memory_percent:.1f}%, GPU:{gpu_percent:.1f}%, GPU Mem:{gpu_memory_percent:.1f}%"

def resource_monitor(pbar, stop_event, pbar_lock):
    while not stop_event.is_set():
        usage = get_resource_usage()
        with pbar_lock:
            pbar.set_postfix_str(usage)
        time.sleep(1)

# Initialize the Segment Anything Model
def initialize_sam():
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "models", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=19,
        pred_iou_thresh=0.50,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=0.7,
        min_mask_region_area=14500,
    )

# Mask processing utilities
def compute_mask_iou(mask1, mask2):
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / (union + 1e-6)

def remove_duplicate_masks(masks, iou_threshold=0.80):
    unique = []
    for m in masks:
        if not any(compute_mask_iou(m['segmentation'], u['segmentation']) > iou_threshold for u in unique):
            unique.append(m)
    return unique

def erode_mask(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    m_uint8 = (mask.astype(np.uint8)) * 255
    eroded = cv2.erode(m_uint8, kernel, iterations=iterations)
    return eroded > 0

def crop_and_save_masks(image, masks, output_folder, erosion_kernel_size=3, erosion_iterations=1):
    for idx, mask in enumerate(masks):
        eroded = erode_mask(mask['segmentation'], kernel_size=erosion_kernel_size, iterations=erosion_iterations)
        x, y, w, h = map(int, mask['bbox'])
        cropped = image[y:y+h, x:x+w]
        m_bool = eroded[y:y+h, x:x+w]
        for c in range(3):
            cropped[:, :, c] = cropped[:, :, c] * m_bool
        out_path = os.path.join(output_folder, f'mask_{idx+1}.png')
        cv2.imwrite(out_path, cropped)

# Download images from URL list
def download_images(file_path):
    input_dir = os.path.join(os.path.dirname(__file__), "input-images")
    os.makedirs(input_dir, exist_ok=True)
    for f in os.listdir(input_dir):
        fp = os.path.join(input_dir, f)
        if os.path.isfile(fp): os.remove(fp)
    urls = []
    if file_path.endswith('.txt'):
        with open(file_path) as f:
            urls = [l.strip() for l in f if l.strip()]
    elif file_path.endswith('.csv'):
        with open(file_path) as f:
            for row in csv.reader(f):
                urls.extend([u.strip() for u in row if u.strip()])
    else:
        print("Unsupported file type.")
        return
    print(f"Found {len(urls)} images.")
    for idx, url in enumerate(tqdm(urls, desc="Downloading images")):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            ext = url.split('.')[-1].split('?')[0]
            if ext.lower() not in ['jpg','jpeg','png','bmp','gif']:
                ext = 'jpg'
            name = os.path.basename(url.split('?')[0]).split('.')[0] or 'image'
            fname = f"{idx+1:04d}_{name}.{ext}"
            with open(os.path.join(input_dir, fname), 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed {url}: {e}")

# Enhance downloaded images for segmentation
def enhance_image():
    input_dir = os.path.join(os.path.dirname(__file__), "input-images")
    output_dir = os.path.join(os.path.dirname(__file__), "enhanced-images")
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp): os.remove(fp)
    imgs = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for img_name in tqdm(imgs, desc="Enhancing images"):
        try:
            with Image.open(os.path.join(input_dir, img_name)) as img:
                img = img.resize((1500, 2250), Image.LANCZOS)
                img = ImageEnhance.Color(img).enhance(1.2)
                img = ImageEnhance.Brightness(img).enhance(0.9)
                img.save(os.path.join(output_dir, img_name))
        except Exception as e:
            print(f"Enhance failed {img_name}: {e}")

# Segment enhanced images using SAM
def segmentation():
    input_dir = os.path.join(os.path.dirname(__file__), "enhanced-images")
    output_dir = os.path.join(os.path.dirname(__file__), "segmented-images")
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not image_files:
        print("No images to segment.")
        return
    pbar_lock = threading.Lock()
    with tqdm(total=len(image_files), desc='Segmenting Images') as pbar:
        stop_event = threading.Event()
        monitor = threading.Thread(target=resource_monitor, args=(pbar, stop_event, pbar_lock))
        monitor.start()
        try:
            mask_generator = initialize_sam()
            for img_file in image_files:
                with pbar_lock:
                    pbar.set_description(f"Segmenting {img_file}")
                img_path = os.path.join(input_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    max_dim = 2250
                    scale = max_dim / max(img.shape[:2])
                    if scale < 1:
                        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                    masks = mask_generator.generate(img)
                    area = img.shape[0]*img.shape[1]
                    masks = [m for m in masks if m['area'] < area * 0.8]
                    masks = remove_duplicate_masks(masks, iou_threshold=0.80)
                    img_name = os.path.splitext(img_file)[0]
                    folder = os.path.join(output_dir, img_name)
                    os.makedirs(folder, exist_ok=True)
                    crop_and_save_masks(img, masks, folder)
                except Exception as e:
                    print(f"Error on {img_file}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()
                with pbar_lock:
                    pbar.update(1)
        finally:
            stop_event.set()
            monitor.join()

# Create a collage of kept images
def create_collage(image_paths, output_path, max_width=2000, background_color=(0, 0, 0)):
    images = []
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error processing image {image_path} for collage: {e}")
    if not images:
        print("No images to create collage.")
        return
    images.sort(key=lambda img: img.size[1])
    rows = []
    current_row = []
    current_width = 0
    max_row_height = 0
    for img in images:
        img_width, img_height = img.size
        if current_width + img_width <= max_width:
            current_row.append(img)
            current_width += img_width
            max_row_height = max(max_row_height, img_height)
        else:
            rows.append((current_row, max_row_height))
            current_row = [img]
            current_width = img_width
            max_row_height = img_height
    if current_row:
        rows.append((current_row, max_row_height))
    collage_width = max_width
    collage_height = sum(height for (_, height) in rows)
    collage_image = Image.new('RGB', (collage_width, collage_height), color=background_color)
    y_offset = 0
    for row_images, row_height in rows:
        x_offset = 0
        for img in row_images:
            collage_image.paste(img, (x_offset, y_offset))
            x_offset += img.size[0]
        y_offset += row_height
    collage_image = collage_image.crop(collage_image.getbbox())
    collage_image.save(output_path)
    print(f"Collage saved to {output_path}")

# Generate per-image folders with enhanced, collage, and segments
# into Desktop/Transcription_Ready_Images/<timestamp>/
def create_transcription_ready_collages():
    base_folder = os.path.join(os.path.dirname(__file__), "segmented-images")
    enhanced_folder = os.path.join(os.path.dirname(__file__), "input-images")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    output_root = os.path.join(desktop, "Transcription_Ready_Images", timestamp)
    os.makedirs(output_root, exist_ok=True)

    # Create a folder for all enhanced images
    all_enhanced_dir = os.path.join(output_root, "Full_Images")
    os.makedirs(all_enhanced_dir, exist_ok=True)
    for enhanced_file_name in os.listdir(enhanced_folder):
        src_enhanced_path = os.path.join(enhanced_folder, enhanced_file_name)
        if os.path.isfile(src_enhanced_path):
            shutil.copy(src_enhanced_path, os.path.join(all_enhanced_dir, enhanced_file_name))
    print(f"Copied all enhanced images to: {all_enhanced_dir}")
    
    # Create a folder for all collages
    collaged_images_dir = os.path.join(output_root, "Collaged_Images")
    os.makedirs(collaged_images_dir, exist_ok=True)

    reader = easyocr.Reader(['en'])
    
    all_collages = []

    for root, dirs, files in os.walk(base_folder):
        if not files:
            continue

        kept_segments = []
        for filename in files:
            if 'segmentation_visualization' in filename or 'collage' in filename:
                continue
            file_path = os.path.join(root, filename)
            try:
                img = Image.open(file_path)
                width, height = img.size
                img_np = np.array(img)
                words = reader.readtext(img_np, detail=0)
                if len(words) < 3 or (width > 800 and height > 1500) or (height >= 4.7 * width or width >= 4.7 * height):
                    # os.remove(file_path) # Decide if you want to remove originals from segmented-images
                    continue
                kept_segments.append(file_path)
            except Exception:
                continue

        if kept_segments:
            folder_name = os.path.basename(root)
            out_dir = os.path.join(output_root, folder_name)
            os.makedirs(out_dir, exist_ok=True)

            # Create and save collage named after folder
            collage_path = os.path.join(out_dir, f"{folder_name}_collage.png")
            create_collage(kept_segments, collage_path)
            
            # Also save a copy to the Collaged_Images folder
            collage_copy_path = os.path.join(collaged_images_dir, f"{folder_name}_collage.png")
            shutil.copy(collage_path, collage_copy_path)
            all_collages.append(collage_copy_path)

            # Copy all kept segments into a subfolder
            segments_dir = os.path.join(out_dir, 'segments')
            os.makedirs(segments_dir, exist_ok=True)
            for seg_path in kept_segments:
                shutil.copy(seg_path, os.path.join(segments_dir, os.path.basename(seg_path)))

            print(f"Created output folder for {folder_name}: {out_dir}")
    
    print(f"All collages saved to: {collaged_images_dir}")

# Main entry point
def main():
    print(80 * "=")
    tprint("Segmentator")
    print(80 * "=")
    print("Welcome To FieldMuseum's Segmentator! v1.1")
    print("Provide a .txt or .csv of image URLs to download and process.")
    file_path = input("Enter path to .txt or .csv: ").strip()
    download_images(file_path)
    print("Download complete.")

    if input("View downloaded images? (Y/N): ").lower() == 'y':
        for f in sorted(os.listdir(os.path.join(os.path.dirname(__file__), "input-images"))):
            print(f)

    if input("Enhance images for segmentation? (Y/N): ").lower() == 'y':
        enhance_image()
        print("Enhancement complete.")

    print(80 * "=")
    if torch.cuda.is_available():
        print("CUDA available:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print("Devices:", torch.cuda.device_count())
        print("Ensure that your device has ATLEAST 8Gb of Video Memory to run this project")
    else:
        print("No GPU available.")

    print(80 * "=")
    if input("Would you like to continue and run Segmentation? (Y/N): ").lower() == "y":
        segmentation()
        print("Segmentation complete. Creating transcription-ready collages...")
        print(80*"=")
        create_transcription_ready_collages()
        print(80*"=")
        print("All Done!! Check your Desktop/Transcription_Ready_Images folder.")
        print("Closing Segmentator...")
    else:
        print("Closing Segmentator...")

if __name__ == "__main__":
    main()
