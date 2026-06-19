from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from diffusionkit.utils import _load_weights


def test_load_weights_mismatch_error():
    module = MagicMock(spec=nn.Module)
    # Simulate a module with 10 parameters
    module.parameters.return_value = [torch.zeros(10)]
    module.named_parameters.return_value = [("param1", torch.zeros(10))]

    with patch("torch.load") as mock_load:
        # Simulate a loaded state_dict with 5 parameters
        mock_load.return_value = {"param1": torch.zeros(5)}

        with pytest.raises(
            ValueError,
            match=r"Total number of parameters in state_dict \(5\) does not match the number of parameters in the module \(10\)",
        ):
            _load_weights(module, "dummy.pt")


def test_load_weights_success():
    module = MagicMock(spec=nn.Module)
    # Simulate a module with 10 parameters
    module.parameters.return_value = [torch.zeros(10)]
    module.named_parameters.return_value = [("param1", torch.zeros(10))]

    with patch("torch.load") as mock_load:
        # Simulate a loaded state_dict with 10 parameters
        mock_load.return_value = {"param1": torch.zeros(10)}

        _load_weights(module, "dummy.pt")

        # Verify load_state_dict was called. In `_load_weights` it's called twice for `.pt` files.
        # But for correctness, let's just make sure it was called with the loaded state dict.
        module.load_state_dict.assert_called_with(mock_load.return_value)
