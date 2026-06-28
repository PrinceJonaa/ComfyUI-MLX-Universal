import os
import uuid
import torch
import numpy as np
import soundfile as sf
import folder_paths

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
