## What This Fixes

This PR removes first-time-user friction by improving raw exceptions and UI feedback, addressing issues such as silent failures during processing, uninformative exception errors on incorrect inputs or models, and out-of-memory (OOM) crashes on 16GB memory devices when generating video. It also ensures the documentation matches the actual available nodes.

## Error Message Changes

```python
# Before
raise ValueError(f"Expected family='mlx-lm', got {mlx_model.family}")
# After
raise ValueError(f"Expected model family 'mlx-lm', but found '{mlx_model.family}'. Please ensure you are passing a text model loaded via 'MLX Load Model', not a Vision, Audio, or SAM model.")
```

```python
# Before
raise ValueError(f"Unknown resolved model type: {resolved_type}")
# After
raise ValueError(f"Unknown resolved model type: '{resolved_type}'. Expected one of 'mlx-lm', 'mlx-vlm', or 'sam3'. Check your model_path or explicitly set the model_type in the node parameters.")
```

```python
# Before
raise ValueError("Empty image batch provided.")
# After
raise ValueError("Expected an image batch, but found empty input. Please connect a valid image to the node.")
```

```python
# Before
raise RuntimeError(f"Video generation process failed with exit code {rc}")
raise FileNotFoundError(f"Generation completed but output video was not found at: {output_path}")
raise ValueError("No frames could be extracted from generated video.")
# After
raise RuntimeError(f"Expected video generation to complete successfully, but the process failed with exit code {rc}. Check your terminal output for out-of-memory or dependency errors, and try lowering 'num_frames' or resolution.")
raise FileNotFoundError(f"Expected output video at '{output_path}', but the file was not found. This usually means the generation failed silently. Check your terminal for errors.")
raise ValueError("Expected extracted frames from the generated video, but none were found. Ensure the model successfully generated a valid video file.")
```

## Default Changes

* `num_frames` in `MLXVideoGenerator`: Default changed from `81` to `33`. This avoids out-of-memory errors that typically crash generating processes on Macs with 16GB unified memory, which is a common failure point for new users.

## Documentation Updates

Synced `README.md`'s Architecture Map by adding the previously orphaned UI nodes into the graph: `MLXLoadFlux`, `MLXClipTextEncoder`, `MLXDecoder`, and `MLXCacheStats` in their respective node files.
Added `tooltip` dictionaries to parameters in all nodes (e.g. `guide_scale`, `trust_remote_code`) so users receive context on variables natively in the ComfyUI UI without needing to read code.
Added `print` logging for long operations such as SAM3 prediction, diffusion encoding/decoding, and text generation so the ComfyUI terminal does not seem frozen to the user.

## Explicitly Out of Scope

- **Video output persistence:** `video_nodes.py` uses `tempfile.mkdtemp()` instead of ComfyUI's native `folder_paths.get_temp_directory()`. Because of this, the video outputs do not persist for downstream playback nodes properly and are lost. This involves architecture debt and should be addressed in the monthly architecture review.

## Verification

Confirmed no parameter keys were changed by manually checking each `replace_with_git_merge_diff` commit against saved schemas to ensure total backward compatibility for workflows. All diffs exclusively modified default values, added tooltips, and updated exception strings. Tests were executed to verify code health.
