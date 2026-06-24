import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from .bridge import pil_to_tensor


def process_sam3_result(
    result, pil_img: Image.Image
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Extracts SAM3 prediction results into ComfyUI-compatible tensors and JSON data.

    Takes the raw `mlx_vlm` SAM3 prediction result and the original PIL image. It generates
    an overlay image with bounding boxes and masks, combines all binary masks into a single
    tensor and a batched tensor, and formats bounding box coordinates into a JSON string.

    Args:
        result: The output from `Sam3Predictor.predict()`, containing masks, boxes, and scores.
        pil_img: The original image as a PIL.Image object.

    Returns:
        tuple containing:
            - out_image (torch.Tensor): The image overlayed with masks and bounding boxes.
            - combined_mask (torch.Tensor): A single unified mask of all detections.
            - individual_masks (torch.Tensor): A batch tensor of individual masks.
            - json_data (str): JSON string containing bounding boxes and confidence scores.
    """
    W, H = pil_img.size
    overlay = pil_img.copy()
    draw_width = max(2, int(min(W, H) * 0.005))

    colors = [
        (255, 0, 0, 100),
        (0, 255, 0, 100),
        (0, 0, 255, 100),
        (255, 255, 0, 100),
        (255, 0, 255, 100),
        (0, 255, 255, 100),
    ]

    num_detections = len(result.scores)
    masks_list = []
    boxes_data = []
    raw_masks = result.masks

    mask_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    draw = ImageDraw.Draw(overlay)

    for i in range(num_detections):
        score = float(result.scores[i])
        box = result.boxes[i]
        x1, y1, x2, y2 = map(float, box)
        mask = np.array(raw_masks[i]).squeeze()
        if mask.shape != (H, W):
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H))
            mask = np.array(mask_pil) > 0
        else:
            mask = mask > 0

        masks_list.append(mask)
        color = colors[i % len(colors)]

        mask_rgba[mask] = color
        draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=draw_width)
        draw.text((x1 + 5, y1 + 5), f"{score:.2f}", fill=color[:3])

        boxes_data.append({"box": [x1, y1, x2, y2], "score": score})

    mask_overlay_pil = Image.fromarray(mask_rgba, mode="RGBA")
    overlay.paste(mask_overlay_pil, (0, 0), mask_overlay_pil)
    out_image = pil_to_tensor(overlay)

    if masks_list:
        combined_mask_np = np.logical_or.reduce(masks_list).astype(np.float32)
        combined_mask = torch.from_numpy(combined_mask_np).unsqueeze(0)
        individual_masks = torch.from_numpy(np.stack(masks_list).astype(np.float32))
    else:
        combined_mask = torch.zeros((1, H, W), dtype=torch.float32)
        individual_masks = torch.zeros((1, H, W), dtype=torch.float32)

    return out_image, combined_mask, individual_masks, json.dumps(boxes_data)
