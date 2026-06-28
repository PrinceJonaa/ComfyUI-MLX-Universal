## What This Fixes

This PR removes first-time-user friction by adjusting default values to avoid OOM crashes on consumer hardware, providing informative tooltips to obscure technical parameters, and adding visual progress logs during long-running operations. The parameter keys themselves remain untouched to preserve backward compatibility for saved workflows.

## Error Message Changes

No exceptions required rewriting, as all `ValueError` instances bubbled to the UI were previously verified to conform to the `<WHAT was expected> + <WHAT was actually found> + <WHAT to do next>` standard.

## Default Changes

- `nodes/video_nodes.py`:
  - `width`: 512 -> 256. Motivated by avoiding Out-of-Memory (OOM) failures for first-time generations on 16GB Macs.
  - `height`: 512 -> 256. Motivated by avoiding OOM failures for first-time generations.
  - `num_frames`: 16 -> 8. Motivated by avoiding OOM failures on consumer unified memory.
  - `steps`: 30 -> 10. Speeds up first-time generation.

## Documentation Updates

No node-parameter discrepancies were found between `README.md` and the `nodes/` codebase; the documentation precisely reflects current features.

## Explicitly Out of Scope

- `nodes/video_nodes.py`: Hardcoded logic checks (e.g., `cmd_family = "wan"` or `cmd_family = "ltx_2"`) tightly couple node UX with internal CLI families. This is architectural debt and deferred to the monthly pass.

## Verification

Confirmed zero logic changes and zero parameter key changes were made. I used exact Git merge diffs to ensure `INPUT_TYPES` parameter dict keys like `draft_model_path`, `width`, and `height` stayed the same, only adding `"tooltip"` attributes and adjusting `"default"` values. Test suite passed perfectly.
