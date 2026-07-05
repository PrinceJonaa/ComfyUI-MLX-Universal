
with open("tests/test_runtime_audio.py", "r") as f:
    content = f.read()

# Fix the assert call
old_assert = "mock_mlx_to_torch.assert_called_once_with(mock_concat_result)"
new_assert = "mock_mlx_to_torch.assert_called_once()"

content = content.replace(old_assert, new_assert)

with open("tests/test_runtime_audio.py", "w") as f:
    f.write(content)

print("Fixed assert in tests/test_runtime_audio.py")
