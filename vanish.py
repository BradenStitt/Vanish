#!/usr/bin/env python3
"""
Vanish - Face anonymization using Florence-2 + SAM2

Detects faces using Florence-2 object detection and segments them with SAM2
for precise pixelation/blurring.

Usage:
    python vanish.py input.jpg output.jpg
    python vanish.py input.mp4 output.mp4
    python vanish.py input.jpg output.jpg --pixel-size 15
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================================
# GLOBALS (loaded lazily)
# ============================================================================
_florence_model = None
_florence_processor = None
_sam2_predictor = None
_device = None


def get_device():
    """Get the best available device."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device


def load_florence():
    """Load Florence-2 model and processor."""
    global _florence_model, _florence_processor
    
    if _florence_model is None:
        print("Loading Florence-2 model...")
        _florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large-ft",
            device_map=str(get_device()),
            trust_remote_code=True,
            torch_dtype=torch.float16 if get_device().type == "cuda" else torch.float32
        )
        _florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large-ft",
            trust_remote_code=True
        )
        print("✓ Florence-2 loaded")
    
    return _florence_model, _florence_processor


def load_sam2(checkpoint_path: str = None, config: str = "sam2_hiera_l.yaml"):
    """Load SAM2 model."""
    global _sam2_predictor
    
    if _sam2_predictor is None:
        print("Loading SAM2 model...")
        
        # Default checkpoint location
        if checkpoint_path is None:
            import os
            cache_dir = os.path.expanduser("~/.cache/sam2/checkpoints")
            os.makedirs(cache_dir, exist_ok=True)
            checkpoint_path = os.path.join(cache_dir, "sam2_hiera_large.pt")
            
            # Download if not exists
            if not os.path.exists(checkpoint_path):
                print("Downloading SAM2 checkpoint...")
                import urllib.request
                url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
                urllib.request.urlretrieve(url, checkpoint_path)
                print("✓ Downloaded")
        
        sam2_model = build_sam2(config, checkpoint_path, device=get_device(), apply_postprocessing=False)
        _sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("✓ SAM2 loaded")
    
    return _sam2_predictor


# ============================================================================
# FACE DETECTION (Florence-2)
# ============================================================================

def find_all_faces(image: Image.Image) -> list:
    """
    Find all human faces in an image using Florence-2 object detection.
    
    Returns:
        List of bounding boxes [[x1, y1, x2, y2], ...]
    """
    model, processor = load_florence()
    
    prompt = "<OD>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(get_device())
    
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048,
            do_sample=False,
        )
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    results = processor.post_process_generation(
        text, task="<OD>", image_size=(image.width, image.height)
    )
    
    faces = []
    for bbox, label in zip(results["<OD>"]["bboxes"], results["<OD>"]["labels"]):
        if label == "human face":
            faces.append(bbox)
    
    return faces


def find_main_speakers(image: Image.Image) -> list:
    """
    Find main speaker faces (to exclude from blurring).
    
    Returns:
        List of bounding boxes for main speakers
    """
    model, processor = load_florence()
    
    prompt = "<CAPTION_TO_PHRASE_GROUNDING> human face (main speaker)"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(get_device())
    
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048,
            do_sample=False,
        )
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    results = processor.post_process_generation(
        text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(image.width, image.height)
    )
    
    speakers = []
    for bbox, label in zip(
        results["<CAPTION_TO_PHRASE_GROUNDING>"]["bboxes"],
        results["<CAPTION_TO_PHRASE_GROUNDING>"]["labels"]
    ):
        if label == "human face":
            speakers.append(bbox)
    
    return speakers


def is_overlapping(box1, box2, threshold=0.7) -> bool:
    """Check if two boxes overlap by at least threshold percentage."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    overlap_area = x_overlap * y_overlap
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    min_area = min(area1, area2)
    
    return overlap_area >= threshold * min_area


def find_passerby_faces(image: Image.Image, exclude_speakers: bool = True) -> list:
    """
    Find all faces except main speakers.
    
    Args:
        image: PIL Image
        exclude_speakers: If True, exclude main speaker faces
    
    Returns:
        List of bounding boxes to blur
    """
    all_faces = find_all_faces(image)
    
    if not exclude_speakers:
        return all_faces
    
    speaker_faces = find_main_speakers(image)
    
    # Filter out faces that overlap with speaker faces
    passerby_faces = [
        face for face in all_faces
        if not any(is_overlapping(face, speaker) for speaker in speaker_faces)
    ]
    
    return passerby_faces


# ============================================================================
# SEGMENTATION (SAM2)
# ============================================================================

def segment_faces(image: Image.Image, bboxes: list) -> np.ndarray:
    """
    Segment faces using SAM2 given bounding boxes.
    
    Returns:
        Array of masks with shape (N, H, W)
    """
    if not bboxes:
        return np.array([])
    
    predictor = load_sam2()
    predictor.set_image(image)
    
    masks, scores, logits = predictor.predict(
        box=bboxes,
        multimask_output=False
    )
    
    masks = np.squeeze(masks)
    
    # Ensure 3D array (N, H, W)
    if masks.ndim == 2:
        masks = np.expand_dims(masks, axis=0)
    
    return masks


# ============================================================================
# PIXELATION
# ============================================================================

def pixelate_region(image: np.ndarray, masks: np.ndarray, pixel_size: int = 10) -> np.ndarray:
    """
    Apply pixelation effect to masked regions.
    
    Args:
        image: Original image (H, W, 3)
        masks: Boolean masks (N, H, W)
        pixel_size: Size of pixel blocks
    
    Returns:
        Pixelated image
    """
    masks = masks.astype(bool)
    height, width = image.shape[:2]
    result = image.copy()
    
    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            block_y_end = min(y + pixel_size, height)
            block_x_end = min(x + pixel_size, width)
            block = image[y:block_y_end, x:block_x_end]
            
            # Combine all masks for this block
            combined_mask = np.zeros(block.shape[:2], dtype=bool)
            for mask in masks:
                block_mask = mask[y:block_y_end, x:block_x_end]
                combined_mask = np.logical_or(combined_mask, block_mask)
            
            if combined_mask.any():
                # Average color of masked pixels
                avg_color = [
                    int(np.mean(channel[combined_mask]))
                    for channel in cv2.split(block)
                ]
                
                # Apply to masked region
                for c in range(3):
                    block[:, :, c][combined_mask] = avg_color[c]
                
                result[y:block_y_end, x:block_x_end] = block
    
    return result


# ============================================================================
# MAIN API
# ============================================================================

def vanish(
    image_path: str,
    output_path: str = None,
    pixel_size: int = 10,
    exclude_speakers: bool = True
) -> np.ndarray:
    """
    Detect and pixelate faces in an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output (optional)
        pixel_size: Pixelation block size
        exclude_speakers: If True, don't blur main speakers
    
    Returns:
        Processed image as numpy array
    """
    image = Image.open(image_path).convert("RGB")
    
    # Find faces to blur
    faces = find_passerby_faces(image, exclude_speakers=exclude_speakers)
    print(f"Found {len(faces)} face(s) to blur")
    
    if not faces:
        result = np.array(image)
    else:
        # Segment faces with SAM2
        masks = segment_faces(image, faces)
        
        # Pixelate
        image_array = np.array(image)
        result = pixelate_region(image_array, masks, pixel_size)
    
    # Save if output path provided
    if output_path:
        Image.fromarray(result).save(output_path)
        print(f"✓ Saved to {output_path}")
    
    return result


def vanish_video(
    input_path: str,
    output_path: str,
    pixel_size: int = 10,
    exclude_speakers: bool = True
) -> None:
    """
    Process a video frame by frame.
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        pixel_size: Pixelation block size
        exclude_speakers: If True, don't blur main speakers
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\rFrame {frame_count}/{total_frames}", end="", flush=True)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        
        # Find and segment faces
        faces = find_passerby_faces(pil_frame, exclude_speakers=exclude_speakers)
        
        if faces:
            masks = segment_faces(pil_frame, faces)
            frame_rgb = pixelate_region(frame_rgb, masks, pixel_size)
        
        # Convert back to BGR and write
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    print()
    cap.release()
    out.release()
    print(f"✓ Saved to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vanish - Face anonymization using Florence-2 + SAM2"
    )
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("output", help="Output path")
    parser.add_argument(
        "--pixel-size", "-p",
        type=int,
        default=10,
        help="Pixelation block size (default: 10)"
    )
    parser.add_argument(
        "--blur-all",
        action="store_true",
        help="Blur all faces including main speakers"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    # Determine if video or image
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    is_video = input_path.suffix.lower() in video_extensions
    
    print(f"Vanish - Face Anonymization")
    print(f"{'='*40}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Pixel size: {args.pixel_size}")
    print(f"Blur all: {args.blur_all}")
    print(f"{'='*40}")
    
    if is_video:
        vanish_video(
            args.input,
            args.output,
            pixel_size=args.pixel_size,
            exclude_speakers=not args.blur_all
        )
    else:
        vanish(
            args.input,
            args.output,
            pixel_size=args.pixel_size,
            exclude_speakers=not args.blur_all
        )


if __name__ == "__main__":
    main()
