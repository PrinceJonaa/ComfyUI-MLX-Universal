## Architectural Thesis

The MLXVideoGenerator node (`nodes/video_nodes.py`) was directly handling heavy MLX background logic, including raw subprocess execution (`subprocess.Popen`), temporary file management, and OpenCV (`cv2`) video reading. This violated the core architectural invariant of maintaining a strict separation between ComfyUI frontend wrappers and the runtime substrate. By extracting this logic into a dedicated `execute_video_generation` function within a new `runtime/video_processing.py` module, we retire this architectural debt. ComfyUI-specific interface objects (like `comfy.utils.ProgressBar`) are now cleanly passed as callbacks, improving fault isolation and enforcing UI independence.

## Debt Location

- `nodes/video_nodes.py`: The `MLXVideoGenerator.generate_video` method, specifically from the `cmd_family` branching down to the final `subprocess.Popen` lifecycle and `cv2` video extraction logic.

## What Changed

- Created `runtime/video_processing.py` containing the `execute_video_generation` function, which orchestrates the subprocess command generation, output tracking, error handling, and video decoding.
- Refactored `MLXVideoGenerator.generate_video` in `nodes/video_nodes.py` to offload execution to this new module, utilizing simple lambda callbacks (`progress_callback`, `progress_absolute_callback`, `interrupt_callback`) to map execution progress back to ComfyUI's internal tracking system without leaking those objects into the runtime layer.

## What Was Not Changed

- The public node interface for `MLXVideoGenerator` remains strictly untouched. All `INPUT_TYPES`, `RETURN_TYPES`, parameter keys, types, and defaults are identical to ensure 100% backward compatibility with serialized user workflows.
- The underlying subprocess generation logic and CLI flags remain completely functionally identical to the previous implementation.

## Backward Compatibility

- Ran syntax verification `py_compile` checks for both the newly created runtime module and the modified node file.
- Ran the core testing suite (`unittest discover tests` and `unittest discover nodes/tests`) which confirmed no regressions were introduced.
- Verified that all callbacks gracefully handle the execution lifecycle as designed.

## Rejected Alternatives

- Leaving the logic inside the node but separating it into a distinct class method was rejected because it still fails the fundamental separation of concerns invariant; runtime processing inherently belongs in the `runtime/` substrate to prevent `nodes/` from becoming a dumping ground for long-running execution logic.

## Follow-On Candidates

- [RM-009] Enforce dict return type hints for INPUT_TYPES: Deferred to keep this PR strictly scoped to the video logic extraction.
