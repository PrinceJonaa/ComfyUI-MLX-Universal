import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from comfyui_mlx_universal.runtime.audio_processing import (
    execute_audio_transcription,
    execute_kokoro_tts,
)


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

    @patch("comfyui_mlx_universal.runtime.model_loader.load_kokoro_pipeline")
    @patch("comfyui_mlx_universal.runtime.bridge.mlx_to_torch")
    @patch("mlx.core.concatenate")
    @patch("mlx.core.array")
    @patch("mlx.core.eval")
    def test_execute_kokoro_tts(
        self, mock_eval, mock_array, mock_concatenate, mock_mlx_to_torch, mock_load
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [("a", "b", [1, 2, 3])]
        mock_load.return_value = mock_pipeline

        mock_tensor = MagicMock()
        mock_tensor.dim.return_value = 1
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_mlx_to_torch.return_value = mock_tensor

        result = execute_kokoro_tts("Hello world!", "af_heart", 1.0)

        self.assertIn("waveform", result)
        self.assertIn("sample_rate", result)
        self.assertEqual(result["sample_rate"], 24000)


if __name__ == "__main__":
    unittest.main()
