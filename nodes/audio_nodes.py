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
                    {
                        "default": "mlx-community/whisper-large-v3-turbo",
                        "tooltip": "Hugging Face model repository ID or local path to load the Whisper model from.",
                    },
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
                "Expected a ComfyUI AUDIO dict format + Found invalid or missing audio input + Connect a valid audio source node"
            )

        # Lazy import of mlx-whisper
        # Prevents ComfyUI from crashing on startup in unsupported environments
        import mlx_whisper
        from ..runtime.registry import make_key

        cache_key = make_key(model_path, "mlx-audio")

        def _loader():
            # load_models() returns a tuple of (model, processor) in older versions or just model if we use mlx_whisper internal cache,
            # but usually it's mlx_whisper.whisper.load_model.
            # To be safe across versions and follow the registry pattern, we wrap it in a dummy loaded object or just load it.
            # mlx-whisper does lazy loading inside `transcribe` but we can force it to be tracked
            # However `mlx_whisper` doesn't expose a clean load function that is independent of transcribe out of the box in some versions.
            # Wait, `mlx_whisper` handles its own caching via `_MODEL_CACHE` internally. But to ensure our registry tracking works
            # (which monitors `len(_MODEL_CACHE)` to evict unified memory), we just track a placeholder to trigger evictions if needed.
            return True

        # Trigger registry tracking
        get_or_load_model(cache_key, _loader)

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
