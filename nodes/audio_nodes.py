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
                        "tooltip": "Hugging Face repository ID for the Whisper model (e.g., 'mlx-community/whisper-large-v3-turbo').",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "MLX Universal/Audio"

    def transcribe(self, audio: dict, model_path: str) -> tuple:
        from ..runtime.audio_processing import execute_audio_transcription

        if (
            not isinstance(audio, dict)
            or "waveform" not in audio
            or "sample_rate" not in audio
        ):
            raise ValueError(
                "Expected ComfyUI AUDIO dict format but found invalid or missing audio input. Connect a valid audio source node to this input."
            )

        text = execute_audio_transcription(audio, model_path)
        return (text,)


class MLXKokoroTTS:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "text": (
                    "STRING",
                    {"multiline": True, "default": "The MLX King lives. Let him cook!"},
                ),
                "voice": (
                    [
                        "af_heart",
                        "af_bella",
                        "af_nicole",
                        "af_sarah",
                        "af_sky",
                        "am_adam",
                        "am_michael",
                        "bf_emma",
                        "bf_isabella",
                        "bm_george",
                        "bm_lewis",
                    ],
                    {"default": "af_heart"},
                ),
                "speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    CATEGORY = "MLX Universal/Audio"

    def generate_audio(self, text: str, voice: str, speed: float) -> tuple:
        from ..runtime.audio_processing import execute_kokoro_tts

        audio_out = execute_kokoro_tts(text, voice, speed)
        return (audio_out,)


NODE_CLASS_MAPPINGS = {
    "MLXWhisperTranscribe": MLXWhisperTranscribe,
    "MLXKokoroTTS": MLXKokoroTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXWhisperTranscribe": "MLX Transcribe Audio (Whisper)",
    "MLXKokoroTTS": "MLX Generate Audio (Kokoro)",
}
