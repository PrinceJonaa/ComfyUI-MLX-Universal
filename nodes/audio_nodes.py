import os
import mlx.core as mx
from ..runtime.registry import get_or_load_model


class MLXAudioTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "mlx-community/whisper-large-v3-mlx"}),
                "audio_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "MLX Universal/Audio"

    def transcribe(self, model_path, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        import mlx_whisper

        def _loader():
            # In MLX-Whisper, model loading and transcription is handled together via transcribe,
            # but we can just use registry to track the string reference or cache logic if needed.
            # mlx_whisper caches internally, but to follow the rules, we just load and track the "state".
            # Currently mlx-whisper doesn't have a split load/generate API that returns a model object easily in public API,
            # wait, actually we can just pass model_path.
            pass

        # Since mlx-whisper handles model loading internally during transcribe,
        # we will mock the registry to just track the memory usage, or we can just use mlx_whisper.transcribe.
        # But wait, registry requires a loader returning an object. Let's return the path as a mock object to track it.
        def _loader():
            # This is a dummy object just so registry tracks it and clears cache if needed.
            return model_path

        get_or_load_model(f"whisper:{model_path}", _loader)

        # Transcribe
        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_path)

        return (result["text"],)


NODE_CLASS_MAPPINGS = {
    "MLXAudioTranscribe": MLXAudioTranscribe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXAudioTranscribe": "MLX Audio Transcribe",
}
