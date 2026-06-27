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

        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except (KeyError, TypeError, AttributeError):
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
                    {
                        "multiline": True,
                        "default": "The MLX King lives. Let him cook!",
                    },
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
        # Lazy import to prevent ComfyUI crashes if not installed
        import numpy as np
        import torch

        from ..runtime.model_loader import load_kokoro_pipeline

        pipeline = load_kokoro_pipeline("prince-canuma/Kokoro-82M")

        print(f"Generating Kokoro TTS audio with voice '{voice}' at {speed}x speed...")
        audio_chunks = []
        for _, _, audio in pipeline(
            text, voice=voice, speed=speed, split_pattern=r"\n+"
        ):
            if len(audio) > 0 and isinstance(audio, (list, tuple)):
                audio_chunks.append(audio[0])
            elif len(audio) > 0:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise ValueError("Kokoro TTS generated no audio for the given text.")

        import mlx.core as mx

        from ..runtime.bridge import mlx_to_torch

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
        return (audio_out,)


NODE_CLASS_MAPPINGS = {
    "MLXWhisperTranscribe": MLXWhisperTranscribe,
    "MLXKokoroTTS": MLXKokoroTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXWhisperTranscribe": "MLX Transcribe Audio (Whisper)",
    "MLXKokoroTTS": "MLX Generate Audio (Kokoro)",
}
