from ..runtime.data_types import LoadedMLXModel
from ..runtime.sam_processing import execute_sam3_prediction


class MLXSAM3Predictor:
    @classmethod
    def INPUT_TYPES(s) -> dict:
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

        out_image, combined_mask, individual_masks, json_data = execute_sam3_prediction(
            mlx_model=mlx_model,
            image=image,
            text_prompt=text_prompt,
            score_threshold=score_threshold,
        )

        return (out_image, combined_mask, individual_masks, json_data)


NODE_CLASS_MAPPINGS = {
    "MLXSAM3Predictor": MLXSAM3Predictor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXSAM3Predictor": "MLX Segment Image",
}
