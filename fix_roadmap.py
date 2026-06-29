with open("roadmap.md", "r") as f:
    content = f.read()

# Resolve git conflict in header
content = content.replace("<<<<<<< ours\n> Last curated: 2026-06-27 at commit e1ee695\n=======\n> Last curated: 2026-06-26 at commit e1ee695\n>>>>>>> theirs", "> Last curated: 2026-06-27 at commit e1ee695")

# Resolve git conflict RM-008 vs RM-005 / RM-009
conflict = """<<<<<<< ours
### [RM-008] Extract video generation subprocess and CV2 logic
- Status: Planned
- Evidence: `nodes/video_nodes.py` directly handles `subprocess.Popen`, temp file management, and `cv2` video reading inside the ComfyUI wrapper class.
- Why it matters: Moves heavy background logic out of UI files into the runtime substrate.
=======
### [RM-005] Standalone Causal Video and Image VAE encode/decode nodes
- Status: Planned
- Evidence: `README.md` claims VAEs in Phase 2 Expansion.
- Why it matters: Achieves the ultimate goal of being the definitive "One-Stop Shop" for all modalities.

### [RM-009] Enforce dict return type hints for INPUT_TYPES
- Status: Planned
- Evidence: Multiple nodes in `nodes/` implement `INPUT_TYPES(s)` without a return type hint (e.g., `-> dict:`), causing static analysis drift.
- Why it matters: Improves strict code cleanliness and static analysis verification for the API.
>>>>>>> theirs"""
content = content.replace(conflict, "")

conflict2 = """<<<<<<< ours
### [RM-009] Enforce dict return type hints for INPUT_TYPES — completed 2026-06-27
- Evidence: Added `-> dict:` to `INPUT_TYPES` methods in `nodes/video_nodes.py`, `nodes/loader_nodes.py`, and `nodes/audio_nodes.py`.

### [RM-005] Standalone Causal Video and Image VAE encode/decode nodes — completed 2026-06-26
- Evidence: Added MLXEncoder to `nodes/diffusion_nodes.py`.
=======
### [RM-008] Extract video generation subprocess and CV2 logic
- Status: Completed
- Evidence: `nodes/video_nodes.py` now uses `execute_video_generation` from `runtime/video_processing.py` to handle `subprocess.Popen`, temp file management, and `cv2` video reading.
>>>>>>> theirs"""

completed_text = """### [RM-009] Enforce dict return type hints for INPUT_TYPES — completed 2026-06-27
- Evidence: Added `-> dict:` to `INPUT_TYPES` methods in `nodes/video_nodes.py`, `nodes/loader_nodes.py`, and `nodes/audio_nodes.py`.

### [RM-008] Extract video generation subprocess and CV2 logic — completed 2026-06-27
- Evidence: `nodes/video_nodes.py` now uses `execute_video_generation` from `runtime/video_processing.py`.

### [RM-005] Standalone Causal Video and Image VAE encode/decode nodes — completed 2026-06-26
- Evidence: Added MLXEncoder to `nodes/diffusion_nodes.py`."""

content = content.replace(conflict2, completed_text)

with open("roadmap.md", "w") as f:
    f.write(content)
