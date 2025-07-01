import os
import shutil
import requests
import csv
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    memory_used = memory.used / (1024 * 1024)  # MB
    memory_total = memory.total / (1024 * 1024)  # MB
    
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_percent = gpu.load * 100
        gpu_memory_used = gpu.memoryUsed
        gpu_memory_total = gpu.memoryTotal
        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        gpu_info = f", GPU:{gpu_percent:.1f}%, GPU Mem:{gpu_memory_percent:.1f}% ({gpu_memory_used:.0f}/{gpu_memory_total:.0f}MB)"
    else:
        gpu_info = ""
    
    return f"CPU:{cpu_percent:.1f}%, Mem:{memory_percent:.1f}% ({memory_used:.0f}/{memory_total:.0f}MB){gpu_info}"

def resource_monitor(pbar, stop_event, pbar_lock):
    while not stop_event.is_set():
        usage = get_resource_usage()
        with pbar_lock:
            pbar.set_postfix_str(usage)
        time.sleep(1)

# Choose between available SAM models
def choose_sam_model():
    print("Available models:")
    print("1. ViT-L (Faster, lower memory usage)")
    print("2. ViT-H (Higher accuracy, requires more memory)")
    print("3. ViT-B (Fastest, lowest memory usage - recommended for systems with limited resources)")
    
    # Check system resources and make a recommendation
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_memory_total = gpu.memoryTotal
        if gpu_memory_total < 4500:  # Less than 4.5GB VRAM
            print("\nRECOMMENDATION: Your GPU has limited memory. Option 3 (ViT-B) is recommended.")
        elif gpu_memory_total < 8000:  # Less than 8GB VRAM
            print("\nRECOMMENDATION: Your GPU has moderate memory. Option 1 (ViT-L) is recommended.")
    
    choice = input("Choose model (1/2/3): ").strip()
    
    if choice == "2":
        return "vit_h", "sam_vit_h_4b8939.pth"
    elif choice == "3":
        return "vit_b", "sam_vit_b_01ec64.pth"
    else:
        return "vit_l", "sam_vit_l_0b3195.pth"

# Initialize the Segment Anything Model
def initialize_sam(model_type=None, model_file=None):
    if model_type is None or model_file is None:
        model_type, model_file = choose_sam_model()
    
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "models", model_file)
    
    # Check available memory and decide on device
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # Get GPU memory info
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_memory_total = gpu.memoryTotal
            print(f"GPU Memory: {gpu_memory_total} MB")
            # If GPU has less than 6GB, use CPU
            if gpu_memory_total < 6000:
                print("GPU memory less than 6GB, using CPU instead")
                use_cuda = False
    
    device = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Adjust parameters based on device
    points_per_side = 16 if device == "cpu" else 24
    crop_n_layers = 0 if device == "cpu" else 1
    
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.90,
        crop_n_layers=crop_n_layers,
        min_mask_region_area=30000,
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
    # Process masks in batches to reduce memory usage
    batch_size = 20
    for i in range(0, len(masks), batch_size):
        batch_masks = masks[i:i+batch_size]
        for idx, mask in enumerate(batch_masks):
            try:
                eroded = erode_mask(mask['segmentation'], kernel_size=erosion_kernel_size, iterations=erosion_iterations)
                x, y, w, h = map(int, mask['bbox'])
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue  # Skip invalid masks
                
                cropped = image[y:y+h, x:x+w].copy()  # Create a copy to avoid modifying original
                m_bool = eroded[y:y+h, x:x+w]
                
                # Check if mask and cropped image have compatible dimensions
                if m_bool.shape[:2] != cropped.shape[:2]:
                    print(f"Skipping mask {i+idx+1} due to dimension mismatch")
                    continue
                
                # Apply mask
                for c in range(3):
                    cropped[:, :, c] = cropped[:, :, c] * m_bool
                
                out_path = os.path.join(output_folder, f'mask_{i+idx+1}.png')
                cv2.imwrite(out_path, cropped)
            except Exception as e:
                print(f"Error processing mask {i+idx+1}: {e}")
        
        # Clear memory after each batch
        gc.collect()

# Process images from local folder
def process_local_images(folder_path):
    input_dir = os.path.join(os.path.dirname(__file__), "input-images")
    os.makedirs(input_dir, exist_ok=True)
    
    # Clear existing images
    for f in os.listdir(input_dir):
        os.remove(os.path.join(input_dir, f))
    
    # Copy images from source folder
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f.lower())[1] in supported_formats]
    
    if not image_files:
        print("No supported image files found in the folder.")
        return
    
    print(f"Found {len(image_files)} images in folder.")
    for idx, img_file in enumerate(tqdm(image_files, desc="Copying images")):
        src_path = os.path.join(folder_path, img_file)
        dst_name = f"{idx+1:04d}_{os.path.splitext(img_file)[0]}{os.path.splitext(img_file)[1]}"
        dst_path = os.path.join(input_dir, dst_name)
        shutil.copy2(src_path, dst_path)

# Download images from URL list
def download_images(file_path):
    input_dir = os.path.join(os.path.dirname(__file__), "input-images")
    os.makedirs(input_dir, exist_ok=True)
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

# Choose input method
def choose_input_method():
    print("\nChoose input method:")
    print("1. Download from URLs (txt/csv file)")
    print("2. Use local image folder")
    
    choice = input("Choose option (1/2): ").strip()
    
    if choice == "2":
        folder_path = input("Enter path to image folder: ").strip()
        if not os.path.exists(folder_path):
            print("Folder does not exist.")
            return False
        process_local_images(folder_path)
        return True
    else:
        file_path = input("Enter path to URL file: ").strip()
        if not os.path.exists(file_path):
            print("File does not exist.")
            return False
        download_images(file_path)
        return True

# Enhance downloaded images for segmentation
def enhance_image():
    input_dir = os.path.join(os.path.dirname(__file__), "input-images")
    output_dir = os.path.join(os.path.dirname(__file__), "enhanced-images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine optimal image size based on available memory
    total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
    if total_memory < 8000:  # Less than 8GB RAM
        target_width, target_height = 1000, 1500
    elif total_memory < 16000:  # Less than 16GB RAM
        target_width, target_height = 1200, 1800
    else:
        target_width, target_height = 1500, 2250
    
    print(f"Resizing images to {target_width}x{target_height} based on available memory")
    
    imgs = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for img_name in tqdm(imgs, desc="Enhancing images"):
        try:
            with Image.open(os.path.join(input_dir, img_name)) as img:
                # Preserve aspect ratio
                width, height = img.size
                ratio = min(target_width/width, target_height/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                img = ImageEnhance.Color(img).enhance(1.2)
                img = ImageEnhance.Brightness(img).enhance(0.9)
                img = ImageEnhance.Contrast(img).enhance(1.1)  # Slightly increase contrast
                img.save(os.path.join(output_dir, img_name))
        except Exception as e:
            print(f"Enhance failed {img_name}: {e}")

# Segment enhanced images using SAM
def segmentation(model_type=None, model_file=None):
    input_dir = os.path.join(os.path.dirname(__file__), "enhanced-images")
    output_dir = os.path.join(os.path.dirname(__file__), "segmented-images")
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    # Sort files by index (assuming filenames start with numbers like 0001_...)
    image_files.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else float('inf'))
    
    if not image_files:
        print("No images to segment.")
        return
    
    # Memory optimization settings
    process_size = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
    print(f"Process memory usage: {process_size:.1f} MB, Total system memory: {total_memory:.1f} MB")
    
    # Determine max image size based on available memory
    max_dim = 1500  # Default for low memory
    if total_memory > 16000:  # More than 16GB RAM
        max_dim = 2250
    elif total_memory > 8000:  # More than 8GB RAM
        max_dim = 1800
    print(f"Using maximum image dimension: {max_dim}")
    
    pbar_lock = threading.Lock()
    with tqdm(total=len(image_files), desc='Segmenting Images') as pbar:
        stop_event = threading.Event()
        monitor = threading.Thread(target=resource_monitor, args=(pbar, stop_event, pbar_lock))
        monitor.start()
        try:
            mask_generator = initialize_sam(model_type, model_file)
            for img_file in image_files:
                with pbar_lock:
                    pbar.set_description(f"Segmenting {img_file}")
                img_path = os.path.join(input_dir, img_file)
                try:
                    # Clear memory before processing each image
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    img = cv2.imread(img_path)
                    scale = max_dim / max(img.shape[:2])
                    if scale < 1:
                        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                    
                    # Process the entire image at once
                    masks = mask_generator.generate(img)
                    
                    area = img.shape[0]*img.shape[1]
                    # Keep masks that are neither too small nor too large
                    min_allowed = area * 0.01
                    max_allowed = area * 0.85
                    masks = [m for m in masks if min_allowed < m['area'] < max_allowed]

                    masks = remove_duplicate_masks(masks, iou_threshold=0.80)
                    img_name = os.path.splitext(img_file)[0]
                    folder = os.path.join(output_dir, img_name)
                    os.makedirs(folder, exist_ok=True)
                    crop_and_save_masks(img, masks, folder)
                except Exception as e:
                    print(f"Error on {img_file}: {e}")
                finally:
                    # Ensure memory is cleared
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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

    # Force CUDA cache clearing and use CPU for EasyOCR
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize EasyOCR with CPU device
    reader = None
    try:
        reader = easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        print("Continuing without OCR text detection...")
        
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
                try:
                    if reader is not None:
                        words = reader.readtext(img_np, detail=0)
                        if len(words) < 3 or (width > 800 and height > 1500) or (height >= 4.7 * width or width >= 4.7 * height):
                            # os.remove(file_path) # Decide if you want to remove originals from segmented-images
                            continue
                    else:
                        # If no OCR reader, just check dimensions
                        if (width > 800 and height > 1500) or (height >= 4.7 * width or width >= 4.7 * height):
                            continue
                    kept_segments.append(file_path)
                except Exception as e:
                    print(f"OCR error on {filename}: {e}")
                    # If OCR fails, still keep the segment if it's a reasonable size
                    if not (width > 800 and height > 1500) and not (height >= 4.7 * width or width >= 4.7 * height):
                        kept_segments.append(file_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
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

# Clean up all image folders for a fresh start
def cleanup_image_folders():
    print("Cleaning up image folders for a fresh start...")
    
    # Clean up input-images folder
    input_dir = os.path.join(os.path.dirname(__file__), "input-images")
    os.makedirs(input_dir, exist_ok=True)
    for f in os.listdir(input_dir):
        fp = os.path.join(input_dir, f)
        if os.path.isfile(fp): 
            os.remove(fp)
    
    # Clean up enhanced-images folder
    enhanced_dir = os.path.join(os.path.dirname(__file__), "enhanced-images")
    os.makedirs(enhanced_dir, exist_ok=True)
    for f in os.listdir(enhanced_dir):
        fp = os.path.join(enhanced_dir, f)
        if os.path.isfile(fp): 
            os.remove(fp)
    
    # Clean up segmented-images folder
    segmented_dir = os.path.join(os.path.dirname(__file__), "segmented-images")
    if os.path.exists(segmented_dir):
        shutil.rmtree(segmented_dir)
    os.makedirs(segmented_dir, exist_ok=True)
    
    print("All image folders cleaned up successfully.")

# Main entry point
def main():
    print(80 * "=")
    tprint("Segmentator")
    print(80 * "=")
    print("Welcome To FieldMuseum's Segmentator! v1.2")
    
    # Clean up all image folders at startup
    cleanup_image_folders()
    
    # Use the choose_input_method function to get images
    if not choose_input_method():
        print("Failed to load images. Exiting.")
        return
    print("Images loaded successfully.")

    if input("View loaded images? (Y/N): ").lower() == 'y':
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
        model_type, model_file = choose_sam_model()
        segmentation(model_type, model_file)
        print("Segmentation complete. Creating transcription-ready collages...")
        print(80*"=")
        # Make sure GPU is fully released before proceeding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all CUDA operations to finish
        
        create_transcription_ready_collages()
        print(80*"=")
        print("All Done!! Check your Desktop/Transcription_Ready_Images folder.")
        print("Closing Segmentator...")
    else:
        print("Closing Segmentator...")

if __name__ == "__main__":
    main()
