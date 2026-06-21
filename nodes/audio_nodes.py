import os
from ..runtime.registry import get_or_load_model, make_key

class MLXWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "model_path": ("STRING", {"default": "mlx-community/whisper-tiny"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "MLX Universal/Audio"

    def transcribe(self, audio_path, model_path):
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Expected a valid audio file path + Could not find file at '{audio_path}' + Please provide a valid file path to the audio_path input.")

        import mlx_whisper
        from mlx_whisper.load_models import load_model

        cache_key = make_key(model_path, "whisper")

        def _load():
            print(f"Loading Whisper model '{model_path}'...")
            return load_model(model_path)

        loaded_model = get_or_load_model(cache_key, _load)

        print(f"Transcribing '{audio_path}'...")
        result = mlx_whisper.transcribe(
            audio_path,
            model=loaded_model,
            path_or_hf_repo=model_path,
        )
        return (result["text"],)

NODE_CLASS_MAPPINGS = {
    "MLXWhisperTranscribe": MLXWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXWhisperTranscribe": "MLX Transcribe Audio",
}
