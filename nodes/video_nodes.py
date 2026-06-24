import comfy.utils
from ..runtime.video_processing import generate_video_subprocess


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
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "num_frames": (
                    "INT",
                    {
                        "default": 33,
                        "min": 1,
                        "max": 500,
                        "tooltip": "Number of frames to generate. Lower this if you run out of unified memory.",
                    },
                ),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
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
        model_repo_or_dir,
        prompt,
        negative_prompt,
        width,
        height,
        num_frames,
        steps,
        guide_scale,
        seed,
        image=None,
        audio_path="",
    ):
        pbar = comfy.utils.ProgressBar(steps)

        def progress_callback(step_diff, step_val):
            pbar.update(step_diff)

        output_path, frames_tensor = generate_video_subprocess(
            model_repo_or_dir=model_repo_or_dir,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guide_scale=guide_scale,
            seed=seed,
            image=image,
            audio_path=audio_path,
            progress_callback=progress_callback
        )

        pbar.update_absolute(steps)
        return (output_path, frames_tensor)


NODE_CLASS_MAPPINGS = {
    "MLXVideoGenerator": MLXVideoGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXVideoGenerator": "MLX Generate Video",
}
