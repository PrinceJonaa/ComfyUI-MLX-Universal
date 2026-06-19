import os
import sys
import re
import tempfile
import subprocess
import shutil
import numpy as np
import torch
import comfy.utils
import comfy.model_management
from ..runtime.bridge import tensor_to_pil


class MLXVideoGenerator:
    @classmethod
    def INPUT_TYPES(s):
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
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 500}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "guide_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.5},
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

        name_lower = model_repo_or_dir.lower()
        if "wan" in name_lower:
            cmd_family = "wan"
        elif "cogvideo" in name_lower:
            cmd_family = "cogvideo"
        else:
            cmd_family = "ltx_2"

        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")

        if cmd_family == "wan":
            cmd = [
                sys.executable,
                "-m",
                "mlx_video.wan_2.generate",
                "--model-dir",
                model_repo_or_dir,
                "--prompt",
                prompt,
                "--negative-prompt",
                negative_prompt,
                "--width",
                str(width),
                "--height",
                str(height),
                "--num-frames",
                str(num_frames),
                "--steps",
                str(steps),
                "--guide-scale",
                str(guide_scale),
                "--output-path",
                output_path,
            ]
            if seed != -1:
                cmd += ["--seed", str(seed)]
            if image is not None:
                pil_imgs = tensor_to_pil(image)
                temp_img_path = os.path.join(temp_dir, "input_frame.png")
                pil_imgs[0].save(temp_img_path)
                cmd += ["--image", temp_img_path]
        elif cmd_family == "cogvideo":
            cmd = [
                sys.executable,
                "-m",
                "mlx_video.cogvideox.generate",
                "--model-dir",
                model_repo_or_dir,
                "--prompt",
                prompt,
                "--width",
                str(width),
                "--height",
                str(height),
                "--num-frames",
                str(num_frames),
                "--steps",
                str(steps),
                "--guidance-scale",
                str(guide_scale),
                "--output-path",
                output_path,
            ]
            if seed != -1:
                cmd += ["--seed", str(seed)]
            if image is not None:
                pil_imgs = tensor_to_pil(image)
                temp_img_path = os.path.join(temp_dir, "input_frame.png")
                pil_imgs[0].save(temp_img_path)
                cmd += ["--image", temp_img_path]
        else:
            cmd = [
                sys.executable,
                "-m",
                "mlx_video.ltx_2.generate",
                "--model-repo",
                model_repo_or_dir,
                "--prompt",
                prompt,
                "--width",
                str(width),
                "--height",
                str(height),
                "--num-frames",
                str(num_frames),
                "--steps",
                str(steps),
                "--cfg-scale",
                str(guide_scale),
                "--output",
                output_path,
            ]
            if seed != -1:
                cmd += ["--seed", str(seed)]
            if image is not None:
                pil_imgs = tensor_to_pil(image)
                temp_img_path = os.path.join(temp_dir, "input_frame.png")
                pil_imgs[0].save(temp_img_path)
                cmd += ["--image", temp_img_path]
            if audio_path and os.path.exists(audio_path):
                cmd += ["--audio-file", audio_path]

        print(f"Running video generation CLI command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        try:
            pbar = comfy.utils.ProgressBar(steps)
            last_step = 0
            buf = ""

            while True:
                comfy.model_management.throw_exception_if_processing_interrupted()
                char = process.stdout.read(1)
                if not char and process.poll() is not None: break
                if char:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    buf += char
                    if char in ('\r', '\n'):
                        match = re.search(r'(\d+)/' + str(steps), buf)
                        if match:
                            try:
                                step_val = int(match.group(1))
                                if step_val > last_step:
                                    pbar.update(step_val - last_step)
                                    last_step = step_val
                            except:
                                pass
                        buf = ""

            pbar.update_absolute(steps)

            rc = process.poll()
            if rc != 0: raise RuntimeError(f"Video generation process failed with exit code {rc}")
            if not os.path.exists(output_path): raise FileNotFoundError(f"Generation completed but output video was not found at: {output_path}")

            import cv2
            cap = cv2.VideoCapture(output_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32) / 255.0)
            cap.release()

            if len(frames) == 0: raise ValueError("No frames could be extracted from generated video.")
            return (output_path, torch.from_numpy(np.stack(frames, axis=0)))
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            shutil.rmtree(temp_dir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "MLXVideoGenerator": MLXVideoGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXVideoGenerator": "MLX Generate Video",
}
