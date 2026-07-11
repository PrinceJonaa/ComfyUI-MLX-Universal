import os
from typing import Any

import mlx.core as mx
import torch

from .bridge import tensor_to_pil
from .data_types import LoadedMLXModel


def execute_text_generation(
    mlx_model: LoadedMLXModel,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    draft_model: Any = None,
    enable_thinking: bool = False,
    thinking_budget: int = 512,
) -> str:
    """
    Executes text generation using mlx-lm.
    This logic has been extracted from the UI nodes to ensure proper separation
    of MLX background processing and ComfyUI interface objects.
    """
    mx.random.seed(seed)
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    tokenizer = mlx_model.processor
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        # tokenize=False ensures the template returns a formatted string instead of token IDs, which mlx_lm.generate expects.
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    # mlx_lm.generate ignores individual kwargs; advanced parameters must be bundled into a sampler object.
    sampler = make_sampler(temp=temperature, top_p=top_p)

    gen_kwargs: dict[str, Any] = {
        "sampler": sampler,
        "max_tokens": max_tokens,
        "verbose": False,
        "enable_thinking": enable_thinking,
        "thinking_budget": thinking_budget,
    }

    if draft_model is not None:
        gen_kwargs["draft_model"] = draft_model

    print(f"Generating text up to {max_tokens} tokens...")
    response = mlx_lm.generate(
        mlx_model.model,
        tokenizer,
        prompt=formatted_prompt,
        **gen_kwargs,
    )
    print("Text generation complete.")
    return response


def execute_image_description(
    mlx_model: LoadedMLXModel,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    enable_thinking: bool,
    thinking_budget: int,
    image: torch.Tensor | None = None,
    audio_path: str = "",
    draft_model: Any = None,
    draft_kind: str = "dflash",
) -> str:
    """
    Executes image description using mlx-vlm.
    This logic has been extracted from the UI nodes to ensure proper separation
    of MLX background processing and ComfyUI interface objects.
    """
    mx.random.seed(seed)
    import mlx_vlm
    from mlx_vlm.prompt_utils import apply_chat_template

    pil_images = tensor_to_pil(image) if image is not None else []
    audios = [audio_path] if audio_path and os.path.exists(audio_path) else []

    formatted_prompt = apply_chat_template(
        mlx_model.processor,
        mlx_model.model.config,
        prompt,
        num_images=len(pil_images),
        num_audios=len(audios),
    )

    gen_kwargs: dict[str, Any] = {
        "temp": temperature,
        "max_tokens": max_tokens,
        "verbose": False,
        "enable_thinking": enable_thinking,
        "thinking_budget": thinking_budget,
    }

    if draft_model is not None:
        gen_kwargs["draft_model"] = draft_model
        gen_kwargs["draft_kind"] = draft_kind

    print(f"Describing image (max {max_tokens} tokens)...")
    response = mlx_vlm.generate(
        mlx_model.model,
        mlx_model.processor,
        formatted_prompt,
        image=pil_images if pil_images else None,
        audio=audios if audios else None,
        **gen_kwargs,
    )
    print("Image description complete.")
    return response


def execute_batch_image_description(
    mlx_model: LoadedMLXModel,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    enable_thinking: bool,
    thinking_budget: int,
    image: torch.Tensor | None = None,
    audio_path: str = "",
    draft_model: Any = None,
    draft_kind: str = "dflash",
) -> tuple:
    """
    Executes batch image description using mlx-vlm.
    Iterates sequentially over a batch of images to prevent OOM errors.
    """
    mx.random.seed(seed)
    import mlx_vlm
    from mlx_vlm.prompt_utils import apply_chat_template

    pil_images = tensor_to_pil(image) if image is not None else []
    audios = [audio_path] if audio_path and os.path.exists(audio_path) else []

    gen_kwargs: dict[str, Any] = {
        "temp": temperature,
        "max_tokens": max_tokens,
        "verbose": False,
        "enable_thinking": enable_thinking,
        "thinking_budget": thinking_budget,
    }

    if draft_model is not None:
        gen_kwargs["draft_model"] = draft_model
        gen_kwargs["draft_kind"] = draft_kind

    if not pil_images:
        # Fallback to standard execution if no images provided
        formatted_prompt = apply_chat_template(
            mlx_model.processor,
            mlx_model.model.config,
            prompt,
            num_images=0,
            num_audios=len(audios),
        )
        print(f"Describing without image (max {max_tokens} tokens)...")
        response = mlx_vlm.generate(
            mlx_model.model,
            mlx_model.processor,
            formatted_prompt,
            image=None,
            audio=audios if audios else None,
            **gen_kwargs,
        )
        return ([response], response)

    text_list = []
    print(
        f"Describing batch of {len(pil_images)} images (max {max_tokens} tokens per image)..."
    )

    for i, pil_img in enumerate(pil_images):
        print(f"Processing image {i + 1}/{len(pil_images)}...")

        formatted_prompt = apply_chat_template(
            mlx_model.processor,
            mlx_model.model.config,
            prompt,
            num_images=1,
            num_audios=len(audios),
        )

        response = mlx_vlm.generate(
            mlx_model.model,
            mlx_model.processor,
            formatted_prompt,
            image=[pil_img],
            audio=audios if audios else None,
            **gen_kwargs,
        )

        text_list.append(response)

        # Explicit evaluation and memory clearing to prevent OOM
        mx.eval()
        mx.metal.clear_cache()

    print("Batch image description complete.")
    concatenated_text = "\n".join(text_list)
    return (text_list, concatenated_text)
