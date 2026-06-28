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

    def transcribe(self, audio: dict, model_path: str) -> tuple:
        from ..runtime.audio_processing import execute_audio_transcription

        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except (KeyError, TypeError, AttributeError):
            raise ValueError(
                "Expected ComfyUI AUDIO dict format + Invalid or missing audio input + Connect a valid audio source node"
            )

        text = execute_audio_transcription(audio, model_path)
        return (text,)


NODE_CLASS_MAPPINGS = {
    "MLXWhisperTranscribe": MLXWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXWhisperTranscribe": "MLX Transcribe Audio (Whisper)",
}
