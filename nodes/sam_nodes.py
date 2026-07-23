from ..runtime.bridge import tensor_to_pil
from ..runtime.data_types import LoadedMLXModel
from ..runtime.sam_processing import process_sam3_result


class MLXSAM3Predictor:
    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {"default": "a dog"}),
                "score_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Minimum confidence score for a prediction to be included. Lower values return more masks.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "mask_batch", "json_data")
    FUNCTION = "predict"
    CATEGORY = "MLX Universal/SAM"

    def predict(
        self,
        mlx_model: LoadedMLXModel,
        image: dict,
        text_prompt: str,
        score_threshold: float,
    ) -> tuple:
        if mlx_model.family != "sam3":
            raise ValueError(
                f"Expected model family 'sam3' but found '{mlx_model.family}'. Please ensure you are passing a SAM model loaded via 'MLX Load Model'."
            )

        from mlx_vlm.models.sam3.generate import Sam3Predictor

        pil_images = tensor_to_pil(image)
        if not pil_images:
            raise ValueError(
                "Expected an image batch but found empty input. Please connect a valid image to the node."
            )

        pil_img = pil_images[0]
        img_w, img_h = pil_img.size

        predictor = Sam3Predictor(
            mlx_model.model, mlx_model.processor, score_threshold=score_threshold
        )
        print(f"Running SAM3 prediction for prompt: '{text_prompt}'...")
        result = predictor.predict(pil_img, text_prompt=text_prompt)
        print(f"SAM3 prediction complete. Found {len(result.scores)} detections.")

        out_image, combined_mask, individual_masks, json_data = process_sam3_result(
            result, pil_img
        )

        return (out_image, combined_mask, individual_masks, json_data)


NODE_CLASS_MAPPINGS = {
    "MLXSAM3Predictor": MLXSAM3Predictor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXSAM3Predictor": "MLX Segment Image",
}
