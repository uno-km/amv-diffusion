"""Mobile FaceDetailer Module for Termux-Diffusion.

Brings ComfyUI Impact Pack / ADetailer capability to mobile on-device diffusion:
1. Detects facial bounding boxes in generated canvas using lightweight on-device vision.
2. Crops and upscales each face to 512x512.
3. Applies high-frequency localized Img2Img refinement (e.g., 2-4 steps, strength 0.35).
4. Blends refined facial features seamlessly back into the canvas with soft feathered alpha masks.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("termux_diffusion.detailer")


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes (l, t, r, b)."""
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2

    inter_l = max(l1, l2)
    inter_t = max(t1, t2)
    inter_r = min(r1, r2)
    inter_b = min(b1, b2)

    inter_w = max(0, inter_r - inter_l)
    inter_h = max(0, inter_b - inter_t)
    inter_area = inter_w * inter_h

    area1 = max(0, r1 - l1) * max(0, b1 - t1)
    area2 = max(0, r2 - l2) * max(0, b2 - t2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def non_max_suppression(
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.35,
) -> List[Tuple[int, int, int, int]]:
    """Merges overlapping candidate bounding boxes using standard Non-Maximum Suppression."""
    if not boxes:
        return []

    # Sort boxes by area descending
    sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    selected: List[Tuple[int, int, int, int]] = []

    for box in sorted_boxes:
        should_keep = True
        for sel in selected:
            if calculate_iou(box, sel) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            selected.append(box)

    return selected


def detect_faces_on_image(
    image: Union[str, Path, Image.Image],
    min_face_size: int = 28,
    max_faces: int = 4,
) -> List[Tuple[int, int, int, int]]:
    """Detects face bounding boxes in an image using available on-device vision detectors.

    Returns:
        List of bounding boxes formatted as (left, top, right, bottom).
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image)).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    w, h = pil_img.size
    np_img = np.array(pil_img)
    candidate_boxes: List[Tuple[int, int, int, int]] = []

    # Priority A: Check termux_vision (official AMEVA vision stack)
    try:
        import termux_vision
        detections = termux_vision.detect_faces(np_img)
        for det in detections:
            bbox = getattr(det, "bbox", det)
            l = getattr(bbox, "left", None)
            t = getattr(bbox, "top", None)
            r = getattr(bbox, "right", None)
            b = getattr(bbox, "bottom", None)
            if None not in (l, t, r, b):
                if (r - l) >= min_face_size and (b - t) >= min_face_size:
                    candidate_boxes.append((int(l), int(t), int(r), int(b)))
    except Exception as e:
        logger.debug("termux_vision face detection skipped or failed: %s", e)

    # Priority B: OpenCV Cascade / YuNet fallback if available
    if not candidate_boxes:
        try:
            import cv2
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
            if os.path.isfile(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_face_size, min_face_size))
                for (x, y, fw, fh) in faces:
                    candidate_boxes.append((int(x), int(y), int(x + fw), int(y + fh)))
        except Exception as cv_err:
            logger.debug("OpenCV face detection skipped or failed: %s", cv_err)

    # Apply NMS to eliminate duplicate clustered candidate boxes
    merged_boxes = non_max_suppression(candidate_boxes, iou_threshold=0.35)

    # Filter boundary margin noise (false detections sticking directly to image borders)
    filtered: List[Tuple[int, int, int, int]] = []
    for (l, t, r, b) in merged_boxes:
        bw, bh = r - l, b - t
        if (l <= 2 or r >= w - 2) and (bw < 40 or bh < 40):
            continue
        filtered.append((l, t, r, b))

    # Prioritize largest face regions and cap to max_faces
    filtered.sort(key=lambda bx: (bx[2] - bx[0]) * (bx[3] - bx[1]), reverse=True)
    final_boxes = filtered[:max_faces] if max_faces > 0 else filtered

    logger.info("[FaceDetailer] Detected %d face region(s) in %dx%d image.", len(final_boxes), w, h)
    return final_boxes


def create_feathered_mask(size: Tuple[int, int], blur_radius: float = 12.0) -> Image.Image:
    """Generates an elliptical alpha mask with smooth feathered Gaussian edges."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Inset by margin to allow smooth edge feathering
    margin_x = max(2, int(w * 0.08))
    margin_y = max(2, int(h * 0.08))
    draw.ellipse([margin_x, margin_y, w - margin_x, h - margin_y], fill=255)

    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    return mask


def refine_face_regions(
    image_path: Union[str, Path],
    generator_func: Callable[..., Any],
    prompt: str,
    output_path: Optional[Union[str, Path]] = None,
    strength: float = 0.35,
    steps: int = 2,
    padding_factor: float = 1.45,
    device: str = "auto",
    model: str = "turbo",
    generator_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Applies ComfyUI-style FaceDetailer: crops, refines with localized Img2Img, and blends back.

    Args:
        image_path: Path to the base generated image.
        generator_func: The termux_diffusion.core.generate callable.
        prompt: Base generation prompt for contextual text conditioning.
        output_path: Destination path for the detailed image (overwrites image_path if None).
        strength: Denoising strength for localized face refinement (default: 0.35).
        steps: Number of sampling steps for the localized face crop (default: 2).
        padding_factor: Expansion factor around face bounding box (default: 1.45).
        device: Target execution device ('auto', 'vulkan', 'cpu', etc.).
        model: Model preset or path.
        generator_kwargs: Additional kwargs passed to generator_func.

    Returns:
        Path to the final detailed image.
    """
    src_path = Path(image_path).resolve()
    dest_path = Path(output_path).resolve() if output_path else src_path

    if not src_path.is_file():
        raise FileNotFoundError(f"[FaceDetailer] Source image file not found: {src_path}")

    base_image = Image.open(src_path).convert("RGB")
    orig_w, orig_h = base_image.size

    boxes = detect_faces_on_image(base_image)
    if not boxes:
        logger.info("[FaceDetailer] No faces detected above threshold. Keeping base canvas intact.")
        if dest_path != src_path:
            base_image.save(dest_path)
        return dest_path

    working_canvas = base_image.copy()
    extra_kwargs = dict(generator_kwargs or {})

    # Strip detailer flags from recursive generator calls to prevent infinite loops
    extra_kwargs.pop("detail_faces", None)
    extra_kwargs.pop("export_gallery", None)

    with tempfile.TemporaryDirectory(prefix="face_detailer_") as tmpdir:
        tmp_dir_path = Path(tmpdir)

        for idx, (l, t, r, b) in enumerate(boxes):
            box_w = r - l
            box_h = b - t
            center_x = l + box_w // 2
            center_y = t + box_h // 2

            # Expand with padding to capture full facial structure, forehead, and chin
            padded_dim = int(max(box_w, box_h) * padding_factor)
            pad_l = max(0, center_x - padded_dim // 2)
            pad_t = max(0, center_y - padded_dim // 2)
            pad_r = min(orig_w, pad_l + padded_dim)
            pad_b = min(orig_h, pad_t + padded_dim)

            # Re-adjust left/top if clamped at right/bottom edges
            actual_w = pad_r - pad_l
            actual_h = pad_b - pad_t
            if actual_w <= 16 or actual_h <= 16:
                continue

            # 1. Crop face region
            face_crop = working_canvas.crop((pad_l, pad_t, pad_r, pad_b))

            # 2. Upscale to 512x512 canvas for high-resolution diffusion pass
            face_upscaled = face_crop.resize((512, 512), Image.Resampling.LANCZOS)
            crop_in_path = tmp_dir_path / f"face_{idx}_in.png"
            crop_out_path = tmp_dir_path / f"face_{idx}_out.png"
            face_upscaled.save(crop_in_path, "PNG")

            # 3. Formulate face-specific prompt
            face_prompt = (
                f"extreme close-up portrait of face, {prompt}, "
                f"hyperdetailed sharp eyes, natural skin micro-texture, authentic facial expression, photorealistic 8k"
            )

            logger.info(
                "[FaceDetailer] Refining face #%d/%d at [%d,%d,%d,%d] (upscaled 512x512, strength=%.2f, steps=%d)...",
                idx + 1, len(boxes), pad_l, pad_t, pad_r, pad_b, strength, steps
            )

            try:
                # 4. Invoke localized Img2Img generation
                generator_func(
                    prompt=face_prompt,
                    init_img=crop_in_path,
                    strength=strength,
                    steps=steps,
                    width=512,
                    height=512,
                    output=crop_out_path,
                    device=device,
                    model=model,
                    export_gallery=False,
                    **extra_kwargs,
                )

                if crop_out_path.is_file():
                    refined_crop_512 = Image.open(crop_out_path).convert("RGB")
                    # 5. Downscale refined face back to native cropped size
                    refined_face = refined_crop_512.resize((actual_w, actual_h), Image.Resampling.LANCZOS)

                    # 6. Apply feathered alpha mask to seamlessly blend back without harsh borders
                    blur_r = max(4.0, actual_w * 0.12)
                    feather_mask = create_feathered_mask((actual_w, actual_h), blur_radius=blur_r)
                    working_canvas.paste(refined_face, (pad_l, pad_t), feather_mask)
                    logger.info("[FaceDetailer] Successfully blended refined face #%d into canvas.", idx + 1)
                else:
                    logger.warning("[FaceDetailer] Refined face #%d output missing; preserving original crop.", idx + 1)
            except Exception as ref_err:
                logger.warning("[FaceDetailer] Localized refinement failed for face #%d: %s. Preserving original.", idx + 1, ref_err)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    working_canvas.save(dest_path, "PNG")
    logger.info("[FaceDetailer] All detected faces refined and merged -> %s", dest_path)
    return dest_path
