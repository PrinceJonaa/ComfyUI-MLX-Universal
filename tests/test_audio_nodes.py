import os
import unittest
from unittest.mock import MagicMock, patch

from tests.test_helper import import_node_module


@unittest.skipIf(
    os.environ.get("REAL_MLX_TESTS") == "1", "Skipping mock tests with real MLX"
)
class TestAudioNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio_nodes = import_node_module("audio_nodes")
        cls.MLXWhisperTranscribe = cls.audio_nodes.MLXWhisperTranscribe
        cls.MLXKokoroTTS = cls.audio_nodes.MLXKokoroTTS

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

    @patch("comfyui_mlx_universal.runtime.audio_processing.execute_audio_transcription")
    def test_transcribe_happy_path(self, mock_execute_audio_transcription):
        mock_execute_audio_transcription.return_value = "hello world"

        node = self.MLXWhisperTranscribe()

        mock_waveform = MagicMock()
        mock_audio = {"waveform": mock_waveform, "sample_rate": 16000}

        result = node.transcribe(mock_audio, "mlx-community/whisper-large-v3-turbo")

        mock_execute_audio_transcription.assert_called_once_with(
            mock_audio, "mlx-community/whisper-large-v3-turbo"
        )
        self.assertEqual(result, ("hello world",))

    @patch("comfyui_mlx_universal.runtime.audio_processing.execute_kokoro_tts")
    def test_kokoro_generate_audio(self, mock_execute_kokoro_tts):
        mock_audio_out = {"waveform": MagicMock(), "sample_rate": 24000}
        mock_execute_kokoro_tts.return_value = mock_audio_out

        node = self.MLXKokoroTTS()
        result = node.generate_audio("test text", "af_heart", 1.0)

        mock_execute_kokoro_tts.assert_called_once_with("test text", "af_heart", 1.0)
        self.assertEqual(result, (mock_audio_out,))


if __name__ == "__main__":
    unittest.main()
