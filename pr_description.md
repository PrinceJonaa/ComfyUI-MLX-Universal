## What was broken
The `execute_video_generation` function in `runtime/video_processing.py` generated temporary image frames during image-to-video generation but lacked cleanup logic. If generation failed or multiple generations were run, these files would accumulate, causing a silent disk space leak.

## How it was discovered
While reviewing the runtime substrate scripts for resource management, I compared `execute_audio_transcription` with `execute_video_generation`. I noticed that audio explicitly deleted its temporary file in a `finally` block, but video processing did not.

## How the fix was verified
I added an `os.remove(temp_img_path)` instruction inside the existing `finally` block for the video generation subprocess. I verified the change by running the local unit tests via `make verify` and observed that the process correctly cleans up temporary image frames without crashing or causing regressions.

## Issues resolved
None specific. Discovered autonomously during a Bug Fixes & Code Health pass.
