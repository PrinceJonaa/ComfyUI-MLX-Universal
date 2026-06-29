## What This Fixes

This PR removes several friction points for first-time users by preventing raw dictionary exceptions from confusing users when inputs are missing, clarifying technical parameter names via tooltips, and tuning video generation defaults to prevent immediate Out-of-Memory (OOM) failures on common hardware setups (like 16GB Apple Silicon Macs).

## Error Message Changes

```python
# Before
try:
    mlx_latent = latent_to_mlx(latent_image)
except (KeyError, TypeError, AttributeError):
    raise ValueError(
        "Expected a valid ComfyUI latent dictionary with a 'samples' tensor + Invalid or missing latent input + Ensure an Empty Latent Image or VAE Encode node is properly connected"
    )

# After
try:
    mlx_latent = latent_to_mlx(latent_image)
except (KeyError, TypeError, AttributeError):
    raise ValueError(
        "Expected a valid ComfyUI latent dictionary with a 'samples' tensor but found an invalid or missing latent input. Ensure an Empty Latent Image or VAE Encode node is properly connected."
    )
```
*(Similar changes applied across audio_nodes.py, diffusion_nodes.py, generate_nodes.py, sam_nodes.py, and bridge.py)*

## Default Changes

- `num_frames` in `MLXVideoGenerator`: **16 -> 8**. Generating 16 frames of video via CLI subprocess is highly likely to trigger an OS-level hard-swap or crash on machines with 16GB unified memory if the user is unaware of the cost. Lowering to 8 ensures a higher likelihood of first-run success.

## Documentation Updates

- Updated the Core Capabilities table in `README.md` to remove the incorrect claim that "Standalone causal video" VAE nodes exist. The codebase currently only supports image VAE encode/decode nodes.
- Verified that all nodes are perfectly mirrored between `nodes/` and the diagram in `README.md`.

## Explicitly Out of Scope

- Removing or migrating `cv2` and `subprocess` logic out of the UI wrapper file `video_nodes.py` into a background worker is out of scope for this UX pass.

## Verification

- Verified all parameter keys remain unchanged (only defaults and `tooltip` metadata dictionaries were updated) to ensure backward compatibility with saved user workflows.
- Diff confirms no execution logic or tensor math was altered.
- All unit tests pass, explicitly testing the newly reformatted exception strings.
