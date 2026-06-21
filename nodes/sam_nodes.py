import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from ..runtime.data_types import LoadedMLXModel
from ..runtime.bridge import tensor_to_pil, pil_to_tensor


class MLXSAM3Predictor:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "mlx_model": ("MLX_MODEL", {"tooltip": "The loaded MLX SAM3 model"}),
                "image": ("IMAGE", {"tooltip": "The input image to segment"}),
                "text_prompt": ("STRING", {"default": "a dog", "tooltip": "Text describing what to segment in the image"}),
                "score_threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Minimum confidence score for a detection to be returned"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "mask_batch", "json_data")
    FUNCTION = "predict"
    CATEGORY = "MLX Universal/SAM"

    def predict(self, mlx_model: LoadedMLXModel, image, text_prompt, score_threshold):
        if mlx_model.family != "sam3":
            raise ValueError(f"Expected a segmentation model (sam3) but got '{mlx_model.family}'. This usually happens if you connect a Text or Vision model to a SAM node. Please connect a model loaded with type 'sam3'.")

        from mlx_vlm.models.sam3.generate import Sam3Predictor

        pil_images = tensor_to_pil(image)
        if not pil_images:
            raise ValueError("Expected an image input but received an empty batch. Check your image loading/routing nodes and ensure they are passing valid image data.")

        pil_img = pil_images[0]
        W, H = pil_img.size

        predictor = Sam3Predictor(
            mlx_model.model, mlx_model.processor, score_threshold=score_threshold
        )
        print(f"Running SAM3 prediction for prompt: '{text_prompt}'...")
        result = predictor.predict(pil_img, text_prompt=text_prompt)
        print("SAM3 prediction completed.")

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

        return (out_image, combined_mask, individual_masks, json.dumps(boxes_data))


NODE_CLASS_MAPPINGS = {
    "MLXSAM3Predictor": MLXSAM3Predictor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXSAM3Predictor": "MLX Segment Image",
}
