import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from diffusionkit.utils import _load_weights, bytes2gigabytes


class TestUtils(unittest.TestCase):
    def test_bytes2gigabytes(self):
        # 0 bytes should be 0 gigabytes
        self.assertEqual(bytes2gigabytes(0), 0)

        # 1 GB in bytes
        one_gb_in_bytes = 1024**3
        self.assertEqual(bytes2gigabytes(one_gb_in_bytes), 1.0)

        # 2.5 GB in bytes
        two_and_half_gb = int(2.5 * 1024**3)
        self.assertEqual(bytes2gigabytes(two_and_half_gb), 2.5)


class TestLoadWeights(unittest.TestCase):
    def test_load_weights_mismatch_error(self):
        module = MagicMock(spec=nn.Module)
        # Simulate a module with 10 parameters
        module.parameters.return_value = [torch.zeros(10)]
        module.named_parameters.return_value = [("param1", torch.zeros(10))]

        with patch("torch.load") as mock_load:
            # Simulate a loaded state_dict with 5 parameters
            mock_load.return_value = {"param1": torch.zeros(5)}

            with self.assertRaises(ValueError) as context:
                _load_weights(module, "dummy.pt")
            self.assertIn(
                "Total number of parameters in state_dict (5) does not match the number of parameters in the module (10)",
                str(context.exception),
            )

    def test_load_weights_success(self):
        module = MagicMock(spec=nn.Module)
        # Simulate a module with 10 parameters
        module.parameters.return_value = [torch.zeros(10)]
        module.named_parameters.return_value = [("param1", torch.zeros(10))]

        with patch("torch.load") as mock_load:
            # Simulate a loaded state_dict with 10 parameters
            mock_load.return_value = {"param1": torch.zeros(10)}

            _load_weights(module, "dummy.pt")

            module.load_state_dict.assert_called_with(mock_load.return_value)


if __name__ == "__main__":
    unittest.main()
