import os
import uuid

import folder_paths
import numpy as np
import soundfile as sf
import torch


def execute_audio_transcription(audio: dict, model_path: str) -> str:
    """
    Executes the audio transcription using mlx-whisper.
    This logic has been extracted from the UI nodes to ensure proper separation
    of MLX background processing and ComfyUI interface objects.
    """
    import mlx_whisper

    from .model_loader import track_audio_model

    # Trigger registry tracking
    track_audio_model(model_path)

    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if isinstance(waveform, torch.Tensor):
        # NOTE: Using index access instead of squeeze(0) to safely handle batches > 1
        audio_np = waveform[0].mean(dim=0).cpu().numpy().astype(np.float32)
    else:
        audio_np = np.array(waveform).astype(np.float32)

    # Use ComfyUI's native temp directory and UUIDs instead of tempfile
    temp_dir = folder_paths.get_temp_directory()
    uid = uuid.uuid4().hex
    tmp_path = os.path.join(temp_dir, f"audio_input_{uid}.wav")

    # Write to a temporary file using soundfile so it handles resampling natively if needed
    sf.write(tmp_path, audio_np, sample_rate)

    try:
        print(f"Transcribing audio using model '{model_path}'...")
        result = mlx_whisper.transcribe(tmp_path, path_or_hf_repo=model_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result.get("text", "").strip()


def execute_kokoro_tts(text: str, voice: str, speed: float) -> dict:
    """
    Executes the audio generation using Kokoro TTS.
    This logic has been extracted from the UI nodes to ensure proper separation
    of MLX background processing and ComfyUI interface objects.
    """
    from .model_loader import load_kokoro_pipeline

    lang_code = voice[:1] if voice else "a"
    pipeline = load_kokoro_pipeline("prince-canuma/Kokoro-82M", lang_code=lang_code)

    print(f"Generating Kokoro TTS audio with voice '{voice}' at {speed}x speed...")
    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        if len(audio) > 0 and isinstance(audio, (list, tuple)):
            audio_chunks.append(audio[0])
        elif len(audio) > 0:
            audio_chunks.append(audio)

    if not audio_chunks:
        raise ValueError("Kokoro TTS generated no audio for the given text.")

    import mlx.core as mx

    from .bridge import mlx_to_torch

    # Convert to mlx arrays if they aren't already, and evaluate explicitly to avoid lazy evaluation traps
    mlx_chunks = [mx.array(c) for c in audio_chunks]
    mx.eval(*mlx_chunks)
    final_audio = mx.concatenate(mlx_chunks, axis=0)
    mx.eval(final_audio)

    # Use bridge to convert to PyTorch efficiently
    final_audio_tensor = mlx_to_torch(final_audio).float()

    # Reshape to [1, channels, samples] for ComfyUI (Kokoro is mono)
    # Note: mlx_to_torch may add a batch dimension for 3D tensors, but for 1D audio it will return 1D
    if final_audio_tensor.dim() == 1:
        final_audio_tensor = final_audio_tensor.unsqueeze(0).unsqueeze(0)
    elif final_audio_tensor.dim() == 2:
        final_audio_tensor = final_audio_tensor.unsqueeze(0)

    audio_out = {"waveform": final_audio_tensor, "sample_rate": 24000}

    print("TTS generation complete.")
    return audio_out
