import os
import tempfile

import soundfile as sf


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

    def transcribe(self, audio: dict, model_path: str) -> tuple[str]:
        if (
            not isinstance(audio, dict)
            or "waveform" not in audio
            or "sample_rate" not in audio
        ):
            raise ValueError(
                "Expected ComfyUI AUDIO dict format but found invalid or missing audio input. Connect a valid audio source node to this input."
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
        import numpy as np
        import torch

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

    def generate_audio(self, text: str, voice: str, speed: float) -> tuple[dict]:
        # Lazy import to prevent ComfyUI crashes if not installed
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
