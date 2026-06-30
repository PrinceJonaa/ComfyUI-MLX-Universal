# The question is: WHY does it throw `ImportError: libmlx.so: cannot open shared object file: No such file or directory` in the CI runner?
# Because the CI runner is macOS arm64, it installed MLX correctly. But WAIT. The error trace says:
# ImportError: libmlx.so: cannot open shared object file: No such file or directory
# Wait, why does MLX on macOS try to load `libmlx.so`?
# OR does some OTHER module try to load it?
# The traceback is:
#   File "/app/.venv/lib/python3.12/site-packages/mlx_whisper/__init__.py", line 3, in <module>
#     from . import audio, decoding, load_models
#   File "/app/.venv/lib/python3.12/site-packages/mlx_whisper/audio.py", line 8, in <module>
#     import mlx.core as mx
# ImportError: libmlx.so: cannot open shared object file: No such file or directory
# Wait, why is it `/app/.venv` in the CI log?
# NO, the CI log says:
# Traceback (most recent call last):
#   File "/home/jules/.pyenv/versions/3.12.13/lib/python3.12/unittest/mock.py", line 1393, in patched
#     with self.decoration_helper(patched,
# ...
#   File "/app/.venv/lib/python3.12/site-packages/mlx_whisper/audio.py", line 8, in <module>
#     import mlx.core as mx
# ImportError: libmlx.so: cannot open shared object file: No such file or directory
# WAIT! The CI log I was looking at earlier was the output of `python3 -m unittest discover tests -v` that I RAN LOCALLY ON THE LINUX SANDBOX (which uses `/app/.venv` and `/home/jules/.pyenv`).
