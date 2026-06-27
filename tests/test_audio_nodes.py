import sys
import unittest
from unittest.mock import MagicMock, patch

from tests.test_helper import import_node_module

class TestAudioNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio_nodes = import_node_module("audio_nodes")
        cls.MLXWhisperTranscribe = cls.audio_nodes.MLXWhisperTranscribe

    def test_transcribe_invalid_audio_format_raises_value_error(self):
        node = self.MLXWhisperTranscribe()
        
        # Test missing dict format
        with self.assertRaises(ValueError) as context:
            node.transcribe("not_a_dict", "mock_whisper_path")
        self.assertIn("Expected ComfyUI AUDIO dict format", str(context.exception))

        # Test missing keys
        with self.assertRaises(ValueError) as context:
            node.transcribe({"sample_rate": 16000}, "mock_whisper_path")
        self.assertIn("Expected ComfyUI AUDIO dict format", str(context.exception))

    @patch("comfyui_mlx_universal.runtime.model_loader.track_audio_model")
    @patch("mlx_whisper.transcribe")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_transcribe_happy_path(self, mock_remove, mock_exists, mock_transcribe, mock_track):
        # Configure transcribe mock
        mock_transcribe.return_value = {"text": "hello world"}

        node = self.MLXWhisperTranscribe()
        
        # Mock audio payload
        mock_waveform = MagicMock()
        mock_audio = {
            "waveform": mock_waveform,
            "sample_rate": 16000
        }

        result = node.transcribe(mock_audio, "mlx-community/whisper-large-v3-turbo")

        mock_track.assert_called_once_with("mlx-community/whisper-large-v3-turbo")
        mock_transcribe.assert_called_once()
        self.assertEqual(result, ("hello world",))
        mock_remove.assert_called_once()


if __name__ == "__main__":
    unittest.main()
