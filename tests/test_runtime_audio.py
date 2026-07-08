import os
import unittest
from unittest.mock import patch

import torch
from comfyui_mlx_universal.runtime.audio_processing import execute_audio_transcription


@unittest.skipIf(
    os.environ.get("REAL_MLX_TESTS") == "1", "Skipping mock tests with real MLX"
)
class TestRuntimeAudio(unittest.TestCase):
    @patch("mlx_whisper.transcribe")
    def test_execute_audio_transcription(self, mock_transcribe):
        mock_transcribe.return_value = {"text": "Hello world!"}

        audio_dict = {"waveform": torch.zeros((1, 1, 16000)), "sample_rate": 16000}

        result = execute_audio_transcription(
            audio_dict, "mlx-community/whisper-large-v3-turbo"
        )
        self.assertEqual(result, "Hello world!")

        # Verify the mock was called correctly
        self.assertTrue(mock_transcribe.called)

        # Extract the args
        args, kwargs = mock_transcribe.call_args
        self.assertTrue(args[0].endswith(".wav"))
        self.assertEqual(
            kwargs["path_or_hf_repo"], "mlx-community/whisper-large-v3-turbo"
        )

    @patch("comfyui_mlx_universal.runtime.bridge.mlx_to_torch")
    @patch("mlx.core.array")
    @patch("mlx.core.concatenate")
    @patch("mlx.core.eval")
    @patch("comfyui_mlx_universal.runtime.model_loader.load_kokoro_pipeline")
    def test_execute_kokoro_tts(
        self, mock_load, mock_eval, mock_concat, mock_array, mock_mlx_to_torch
    ):
        # Setup mock pipeline
        mock_pipeline_instance = mock_load.return_value

        # Generator yielding one tuple
        def mock_generator(*args, **kwargs):
            yield (None, None, [1, 2, 3])

        mock_pipeline_instance.side_effect = mock_generator

        # Mock MLX array and bridge
        mock_array.return_value = "mlx_array"
        mock_concat.return_value = "concatenated_mlx_array"

        mock_tensor = torch.zeros((1, 1, 10))
        mock_mlx_to_torch.return_value = mock_tensor

        from comfyui_mlx_universal.runtime.audio_processing import execute_kokoro_tts

        result = execute_kokoro_tts("test", "af_heart", 1.0)

        self.assertIn("waveform", result)
        self.assertIn("sample_rate", result)

        mock_load.assert_called_once()
        mock_pipeline_instance.assert_called_once()
        mock_mlx_to_torch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
