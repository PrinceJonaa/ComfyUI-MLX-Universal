import os
import mlx.core as mx
from ..runtime.data_types import LoadedMLXModel
from ..runtime.bridge import tensor_to_pil
from ..runtime.registry import get_or_load_draft_model, make_key


class MLXLMGenerateText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Explain quantum computing in simple terms.",
                    },
                ),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 16384}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "MLX Universal/LM"

    def generate(
        self, mlx_model: LoadedMLXModel, prompt, max_tokens, temperature, top_p, seed
    ):
        if mlx_model.family != "mlx-lm":
            raise ValueError(f"Expected family='mlx-lm', got {mlx_model.family}")

        mx.random.seed(seed)
        import mlx_lm

        tokenizer = mlx_model.processor
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt

        response = mlx_lm.generate(
            mlx_model.model,
            tokenizer,
            prompt=formatted_prompt,
            temp=temperature,
            max_tokens=max_tokens,
            verbose=False,
        )
        return (response,)


class MLXVLMDescribeImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe this image in detail."},
                ),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 16384}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "thinking_budget": ("INT", {"default": 512, "min": 0, "max": 8192}),
            },
            "optional": {
                "image": ("IMAGE",),
                "audio_path": ("STRING", {"default": ""}),
                "draft_model_path": ("STRING", {"default": ""}),
                "draft_kind": (["dflash", "eagle3", "mtp"], {"default": "dflash"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "MLX Universal/VLM"

    def run(
        self,
        mlx_model: LoadedMLXModel,
        prompt,
        max_tokens,
        temperature,
        seed,
        enable_thinking,
        thinking_budget,
        image=None,
        audio_path="",
        draft_model_path="",
        draft_kind="dflash",
    ):

        if mlx_model.family != "mlx-vlm":
            raise ValueError(f"Expected family='mlx-vlm', got {mlx_model.family}")

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

        gen_kwargs = {
            "temp": temperature,
            "max_tokens": max_tokens,
            "verbose": False,
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget,
        }

        if draft_model_path:
            from mlx_vlm.speculative.drafters import load_drafter

            draft_key = make_key(draft_model_path, "draft")

            def _load_draft():
                print(f"Loading draft model '{draft_model_path}'...")
                return load_drafter(draft_model_path)

            draft_model, draft_kind_res = get_or_load_draft_model(
                draft_key, _load_draft
            )
            gen_kwargs["draft_model"] = draft_model
            gen_kwargs["draft_kind"] = draft_kind_res

        response = mlx_vlm.generate(
            mlx_model.model,
            mlx_model.processor,
            formatted_prompt,
            image=pil_images if pil_images else None,
            audio=audios if audios else None,
            **gen_kwargs,
        )
        return (response,)


NODE_CLASS_MAPPINGS = {
    "MLXLMGenerateText": MLXLMGenerateText,
    "MLXVLMDescribeImage": MLXVLMDescribeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXLMGenerateText": "MLX Generate Text",
    "MLXVLMDescribeImage": "MLX Understand Image",
}
