#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

MODEL_REPO_IDS = {
    "argmaxinc/mlx-stable-diffusion-3-medium": "argmaxinc/mlx-stable-diffusion-3-medium",
    "sd3-8b-unreleased": "models/sd3_8b_beta.safetensors",  # unreleased
    "argmaxinc/mlx-FLUX.1-schnell": "argmaxinc/mlx-FLUX.1-schnell",
    "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
    "argmaxinc/mlx-FLUX.1-dev": "argmaxinc/mlx-FLUX.1-dev",
}

T5_MAX_LENGTH = {
    "argmaxinc/mlx-stable-diffusion-3-medium": 512,
    "argmaxinc/mlx-FLUX.1-schnell": 256,
    "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": 256,
    "argmaxinc/mlx-FLUX.1-dev": 512,
}
