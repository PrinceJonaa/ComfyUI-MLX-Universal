## What This Fixes

First-time users were encountering vague Python exceptions (`ValueError`) with no actionable guidance when connecting incorrect model types to nodes or providing empty inputs. Furthermore, many critical parameters lacked tooltips, requiring users to read the source code to understand their purpose, and the documentation in `README.md` lagged behind actual node capabilities. This PR removes those friction points by implementing standardized, actionable error messages, adding human-readable tooltips to previously undocumented parameters, and syncing the core documentation.

## Error Message Changes

```python
# Before (generate_nodes.py)
raise ValueError(f"Expected model family 'mlx-lm' but found '{mlx_model.family}'. Please ensure you are passing a text model loaded via 'MLX Load Model', not a Vision, Audio, or SAM model.")

# After
raise ValueError(f"Expected model family 'mlx-lm', but found '{mlx_model.family}'. Please ensure you are passing a text model loaded via 'MLX Load Model', not a Vision, Audio, or SAM model.")
```

```python
# Before (loader_nodes.py)
raise ValueError("Draft model path cannot be empty.")

# After
raise ValueError("Expected a valid Hugging Face repository ID or local path, but found an empty draft model path. Please provide a valid model path for the draft model.")
```

```python
# Before (sam_nodes.py)
raise ValueError("Expected an image batch but found empty input. Please connect a valid image to the node.")

# After
raise ValueError("Expected an image batch, but found an empty input. Please connect a valid image to the node.")
```

```python
# Before (audio_nodes.py)
raise ValueError("Kokoro TTS generated no audio for the given text.")

# After
raise ValueError("Expected generated audio from Kokoro TTS, but found no audio for the given text. Please provide a different text prompt.")
```

## Default Changes

No defaults were changed in this PR.

## Documentation Updates

- Updated the "Core Capabilities" table in `README.md` to include Kokoro TTS under the Audio section.
- Added a new row for `Embeddings` (`mlx-embeddings`) to the "Core Capabilities" table in `README.md`.
- Synchronized the Mermaid architecture diagram in `README.md` to accurately reflect the existence and routing of `MLX Generate Text Embedding`, `MLX Generate Audio (Kokoro)`, and `MLX Load Draft Model`.

## Explicitly Out of Scope

- Consolidating the validation logic for ComfyUI dictionary formats across different modalities. Currently, validation happens inside the node wrappers (e.g., checking for `"waveform"` and `"sample_rate"` in `audio_nodes.py`).

## Verification

- Verified no parameter keys were renamed across any node (only `tooltip` fields were appended to existing dictionaries).
- Verified that all unit tests pass globally via `make verify`.
- Successfully ran `python3 -m unittest discover tests` and `python3 -m unittest discover nodes/tests` to confirm backward compatibility.
