import os
import sys
import folder_paths
import comfy.utils
import comfy.model_management
from ..runtime.video_processing import execute_video_generation


class MLXVideoGenerator:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "model_repo_or_dir": (
                    "STRING",
                    {"default": "mlx-community/LTX-2-dev-bf16"},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Two dogs sitting on a beach wearing sunglasses, close up, cinematic, sunset",
                    },
                ),
                "negative_prompt": ("STRING", {"default": "blurry, low quality"}),
                "width": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Resolution. Lower this if you run out of unified memory.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Resolution. Lower this if you run out of unified memory.",
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 500,
                        "tooltip": "Number of frames to generate. Lower this if you run out of unified memory.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 200,
                        "tooltip": "Number of diffusion steps. Higher values take longer but improve quality.",
                    },
                ),
                "guide_scale": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "tooltip": "Classifier-Free Guidance (CFG) scale. Higher values closely follow the prompt but may introduce artifacts.",
                    },
                ),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "audio_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "images")
    FUNCTION = "generate_video"
    CATEGORY = "MLX Universal/Video"

    def generate_video(
        self,
        model_repo_or_dir: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        guide_scale: float,
        seed: int,
        image: dict | None = None,
        audio_path: str = "",
    ) -> tuple:

        name_lower = model_repo_or_dir.lower()
        if "wan" in name_lower:
            cmd_family = "wan"
        elif "cogvideo" in name_lower:
            cmd_family = "cogvideo"
        else:
            cmd_family = "ltx_2"

        temp_dir = folder_paths.get_temp_directory()
        pbar = comfy.utils.ProgressBar(steps)

        def progress_callback(step_val):
            pbar.update(step_val)

        def progress_absolute_callback(step_val):
            pbar.update_absolute(step_val)

        def interrupt_callback():
            comfy.model_management.throw_exception_if_processing_interrupted()

        return execute_video_generation(
            model_repo_or_dir=model_repo_or_dir,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guide_scale=guide_scale,
            seed=seed,
            temp_dir=temp_dir,
            image=image,
            audio_path=audio_path,
            progress_callback=progress_callback,
            progress_absolute_callback=progress_absolute_callback,
            interrupt_callback=interrupt_callback,
        )


NODE_CLASS_MAPPINGS = {
    "MLXVideoGenerator": MLXVideoGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXVideoGenerator": "MLX Generate Video",
}
