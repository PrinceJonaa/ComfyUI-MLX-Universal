from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LoadedMLXModel:
    family: str
    model_path: str
    model_type: str
    trust_remote_code: bool
    quantize_activations: bool
    model: Any
    processor: Optional[Any] = None
