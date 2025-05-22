import os
import requests
import csv
from art import *
from tqdm import tqdm
from PIL import Image, ImageEnhance
import psutil
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
gc = __import__('gc')
import GPUtil
import threading
import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.ops import boxes

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
    sam_checkpoint = os.path.expanduser("c:\\Users\\Riley\\Desktop\\sam_vit_h_4b8939.pth")  # Update path as needed
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=19,
        pred_iou_thresh=0.90,
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

# Visualization and cropping

#Do we really need the visualization?

# def visualize_and_save_segmentation(image, masks, output_folder):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     for mask in masks:
#         plt.contour(mask['segmentation'], colors="red")
#     plt.axis('off')
#     plt.tight_layout()
#     out_file = os.path.join(output_folder, 'visualization.png')
#     plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
#     plt.close()


def crop_and_save_masks(image, masks, output_folder, erosion_kernel_size=3, erosion_iterations=1):
    for idx, mask in enumerate(masks):
        mask_img = mask['segmentation']
        eroded = erode_mask(mask_img, kernel_size=erosion_kernel_size, iterations=erosion_iterations)
        x, y, w, h = map(int, mask['bbox'])
        cropped = image[y:y+h, x:x+w]
        m_bool = eroded[y:y+h, x:x+w]
        for c in range(3):
            cropped[:, :, c] = cropped[:, :, c] * m_bool
        out_path = os.path.join(output_folder, f'mask_{idx+1}.png')
        cv2.imwrite(out_path, cropped)
        #print(f"Saved {out_path}")

# Original download and enhancement functions

def download_images(file_path):
    input_dir = os.path.join(os.path.dirname(__file__), "Input-Images")
    os.makedirs(input_dir, exist_ok=True)
    # clear directory
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


def enhance_image():
    input_dir = os.path.join(os.path.dirname(__file__), "Input-Images")
    output_dir = os.path.join(os.path.dirname(__file__), "enhanced-images")
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp): os.remove(fp)
    imgs = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for img_name in tqdm(imgs, desc="Enhancing images"):
        try:
            with Image.open(os.path.join(input_dir, img_name)) as img:
                # Resize to 1500x2250 (width x height)
                img = img.resize((1500, 2250), Image.LANCZOS)
                img = ImageEnhance.Color(img).enhance(1.2)
                img = ImageEnhance.Brightness(img).enhance(0.9)
                img.save(os.path.join(output_dir, img_name))
        except Exception as e:
            print(f"Enhance failed {img_name}: {e}")

# Segmentation function integrating SAM logic

def segmentation():
    input_dir = os.path.join(os.path.dirname(__file__), "enhanced-images")
    output_dir = os.path.join(os.path.dirname(__file__), "segmented-images")
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not image_files:
        print("No images to segment.")
        return
    # Resource monitoring setup
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
                    # resize if needed
                    max_dim = 2250
                    scale = max_dim / max(img.shape[:2])
                    if scale < 1:
                        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                    masks = mask_generator.generate(img)
                    # filter and dedupe
                    area = img.shape[0]*img.shape[1]
                    masks = [m for m in masks if m['area'] < area * 0.8]
                    masks = remove_duplicate_masks(masks, iou_threshold=0.80)
                    # save results
                    img_name = os.path.splitext(img_file)[0]
                    folder = os.path.join(output_dir, img_name)
                    os.makedirs(folder, exist_ok=True)
                    #visualize_and_save_segmentation(img, masks, folder)
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

# Main entry point

def main():
    print(80*"=")
    tprint("Segmentator")
    print(80*"=")
    print("Welcome To FieldMuseum's Segmentator! v1.0")
    print("Provide a .txt or .csv of image URLs to download and process.")
    file_path = input("Enter path to .txt or .csv: ").strip()
    download_images(file_path)
    print("Download complete.")
    if input("View downloaded images? (Y/N): ").lower() == 'y':
        for f in sorted(os.listdir(os.path.join(os.path.dirname(__file__), "Input-Images"))):
            print(f)
    if input("Enhance images for segmentation? (Y/N): ").lower() == 'y':
        enhance_image()
        print("Enhancement complete.")
    # Show capabilities
    print(80*"=")
    if torch.cuda.is_available():
        print("CUDA available:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print("Devices:", torch.cuda.device_count())
    else:
        print("No GPU available.")
    print(80*"=")
    print("Running segmentation")
    
    run_segmentation = input("Would you like to continue and run Segmentation? (Y/N): ")
    if run_segmentation.lower() == "y":
        segmentation()
        print("All finished... Closing Segmentator")
    else:
        print("Closing Segmentator...")


    

if __name__ == "__main__":
    main()


## c:\Users\Riley\Desktop\10URL.txt