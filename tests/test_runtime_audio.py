import unittest
from unittest.mock import patch

import torch
from comfyui_mlx_universal.runtime.audio_processing import execute_audio_transcription


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


if __name__ == "__main__":
    unittest.main()
