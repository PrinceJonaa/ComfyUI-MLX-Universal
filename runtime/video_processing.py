import os
import sys
import re
import uuid
import subprocess
import numpy as np
import torch
import folder_paths
import comfy.utils
import comfy.model_management
from .bridge import tensor_to_pil

def generate_video_subprocess(
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
    progress_callback=None
):
    """
    Executes the MLX video generation subprocess and extracts the resulting frames.
    """
    name_lower = model_repo_or_dir.lower()
    if "wan" in name_lower:
        cmd_family = "wan"
    elif "cogvideo" in name_lower:
        cmd_family = "cogvideo"
    else:
        cmd_family = "ltx_2"

    temp_dir = folder_paths.get_temp_directory()
    uid = uuid.uuid4().hex
    output_path = os.path.join(temp_dir, f"output_{uid}.mp4")
    temp_img_name = f"input_frame_{uid}.png"

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
            temp_img_path = os.path.join(temp_dir, temp_img_name)
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
            temp_img_path = os.path.join(temp_dir, temp_img_name)
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
            temp_img_path = os.path.join(temp_dir, temp_img_name)
            pil_imgs[0].save(temp_img_path)
            cmd += ["--image", temp_img_path]
        if audio_path and os.path.exists(audio_path):
            cmd += ["--audio-file", audio_path]

    print(f"Running video generation CLI command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    try:
        last_step = 0
        buf = ""

        while True:
            comfy.model_management.throw_exception_if_processing_interrupted()
            char = process.stdout.read(1)
            if not char and process.poll() is not None:
                break
            if char:
                sys.stdout.write(char)
                sys.stdout.flush()
                buf += char
                if char in ("\r", "\n"):
                    match = re.search(r"(\d+)/" + str(steps), buf)
                    if match:
                        try:
                            step_val = int(match.group(1))
                            if step_val > last_step:
                                if progress_callback:
                                    progress_callback(step_val - last_step, step_val)
                                last_step = step_val
                        except Exception:
                            pass
                    buf = ""

        rc = process.poll()
        if rc != 0:
            raise RuntimeError(
                f"Expected video generation to complete successfully, but the process failed with exit code {rc}. Check your terminal output for out-of-memory or dependency errors, and try lowering 'num_frames' or resolution."
            )
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"Expected output video at '{output_path}', but the file was not found. This usually means the generation failed silently. Check your terminal for errors."
            )

        import cv2

        cap = cv2.VideoCapture(output_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()

        if len(frames) == 0:
            raise ValueError(
                "Expected extracted frames from the generated video, but none were found. Ensure the model successfully generated a valid video file."
            )
        return (output_path, torch.from_numpy(np.stack(frames, axis=0)))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
