import os
import re
import subprocess
import sys
import uuid
from typing import Callable, Optional

import numpy as np
import torch

from .bridge import tensor_to_pil


def _build_video_cmd(
    cmd_family: str,
    model_repo_or_dir: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    guide_scale: float,
    seed: int,
    output_path: str,
    image,
    temp_img_path: str,
    audio_path: str,
) -> list[str]:
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
        pil_imgs[0].save(temp_img_path)
        cmd += ["--image", temp_img_path]
    if cmd_family == "ltx_2" and audio_path and os.path.exists(audio_path):
        cmd += ["--audio-file", audio_path]

    return cmd


def execute_video_generation(
    model_repo_or_dir: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    guide_scale: float,
    seed: int,
    temp_dir: str,
    image=None,
    audio_path: str = "",
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_absolute_callback: Optional[Callable[[int], None]] = None,
    interrupt_callback: Optional[Callable[[], None]] = None,
) -> tuple[str, torch.Tensor]:
    """
    Executes the video generation subprocess and processes the result.
    This logic has been extracted from the UI nodes to ensure proper separation
    of MLX background processing and ComfyUI interface objects.
    """
    name_lower = model_repo_or_dir.lower()
    if "wan" in name_lower:
        cmd_family = "wan"
    elif "cogvideo" in name_lower:
        cmd_family = "cogvideo"
    else:
        cmd_family = "ltx_2"

    uid = uuid.uuid4().hex
    output_path = os.path.join(temp_dir, f"output_{uid}.mp4")
    temp_img_name = f"input_frame_{uid}.png"
    temp_img_path = os.path.join(temp_dir, temp_img_name)

    cmd = _build_video_cmd(
        cmd_family,
        model_repo_or_dir,
        prompt,
        negative_prompt,
        width,
        height,
        num_frames,
        steps,
        guide_scale,
        seed,
        output_path,
        image,
        temp_img_path,
        audio_path,
    )

    print(f"Running video generation CLI command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    try:
        last_step = 0
        buf = ""

        while True:
            # Required for long-running subprocesses so user interruption doesn't leave orphaned generator processes
            if interrupt_callback:
                interrupt_callback()

            char = process.stdout.read(1) if process.stdout is not None else ""
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
                                    progress_callback(step_val - last_step)
                                last_step = step_val
                        except Exception:
                            pass
                    buf = ""

        if progress_absolute_callback:
            progress_absolute_callback(steps)

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
