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

    @patch("comfyui_mlx_universal.runtime.model_loader.load_kokoro_pipeline")
    @patch("mlx.core.array")
    @patch("mlx.core.concatenate")
    @patch("mlx.core.eval")
    @patch("comfyui_mlx_universal.runtime.bridge.mlx_to_torch")
    def test_execute_kokoro_tts(
        self, mock_mlx_to_torch, mock_eval, mock_concat, mock_array, mock_load
    ):
        # Setup mock pipeline
        mock_pipeline = unittest.mock.MagicMock()
        mock_load.return_value = mock_pipeline

        # Setup mock generation output: generator yields (text, phonemes, audio)
        mock_audio_chunk = [1, 2, 3]
        mock_pipeline.return_value = [("test", "test", mock_audio_chunk)]

        # Setup MLX and Torch mocks
        mock_mlx_array = unittest.mock.MagicMock()
        mock_array.return_value = mock_mlx_array

        mock_concat_result = unittest.mock.MagicMock()
        mock_concat.return_value = mock_concat_result

        mock_torch_tensor = unittest.mock.MagicMock()
        mock_torch_tensor.dim.return_value = 1
        # When unsqeezed, return itself for chaining
        mock_torch_tensor.unsqueeze.return_value = mock_torch_tensor
        mock_torch_tensor.float.return_value = mock_torch_tensor
        mock_mlx_to_torch.return_value = mock_torch_tensor

        from comfyui_mlx_universal.runtime.audio_processing import execute_kokoro_tts

        result = execute_kokoro_tts("Hello", "af_heart", 1.0)

        self.assertIn("waveform", result)
        self.assertIn("sample_rate", result)
        self.assertEqual(result["sample_rate"], 24000)

        mock_load.assert_called_once_with("prince-canuma/Kokoro-82M")
        mock_pipeline.assert_called_once_with(
            "Hello", voice="af_heart", speed=1.0, split_pattern=r"\n+"
        )
        mock_mlx_to_torch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
