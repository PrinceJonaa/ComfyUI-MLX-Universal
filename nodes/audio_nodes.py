import os
import tempfile
import soundfile as sf
from ..runtime.registry import get_or_load_model


class MLXWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_path": (
                    "STRING",
                    {"default": "mlx-community/whisper-large-v3-turbo"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "MLX Universal/Audio"

    def transcribe(self, audio, model_path):
        if (
            not isinstance(audio, dict)
            or "waveform" not in audio
            or "sample_rate" not in audio
        ):
            raise ValueError(
                "Expected ComfyUI AUDIO dict format + Invalid or missing audio input + Connect a valid audio source node"
            )

        # Lazy import of mlx-whisper
        # Prevents ComfyUI from crashing on startup in unsupported environments
        import mlx_whisper
        from ..runtime.model_loader import track_audio_model

        # Trigger registry tracking
        track_audio_model(model_path)

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # The waveform is usually a tensor [batch, channels, samples].
        # Convert it to mono numpy array for mlx-whisper.
        import torch
        import numpy as np

        if isinstance(waveform, torch.Tensor):
            # NOTE: Using squeeze(0) assumes batch size 1; a batch size >1 will fail to reduce the dimension (see RM-011).
            # Take the first batch and mean across channels
            audio_np = waveform[0].mean(dim=0).cpu().numpy().astype(np.float32)
        else:
            audio_np = np.array(waveform).astype(np.float32)

        # mlx-whisper expects 16kHz audio or a file path
        # Write to a temporary file using soundfile so it handles resampling natively if needed
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_np, sample_rate)
            tmp_path = tmp_file.name

        try:
            print(f"Transcribing audio using model '{model_path}'...")
            result = mlx_whisper.transcribe(tmp_path, path_or_hf_repo=model_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        text = result.get("text", "").strip()
        return (text,)


NODE_CLASS_MAPPINGS = {
    "MLXWhisperTranscribe": MLXWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXWhisperTranscribe": "MLX Transcribe Audio (Whisper)",
}
