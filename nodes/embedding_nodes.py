import torch


class MLXTextEmbedding:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "I like reading",
                        "tooltip": "The text string to generate embeddings for.",
                    },
                ),
                "model_path": (
                    "STRING",
                    {
                        "default": "mlx-community/all-MiniLM-L6-v2-4bit",
                        "tooltip": "Hugging Face repository ID for the embedding model.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("embedding",)
    FUNCTION = "generate_embedding"
    CATEGORY = "MLX Universal/Embeddings"

    def generate_embedding(self, text: str, model_path: str) -> tuple[torch.Tensor]:
        from ..runtime.embedding_processing import generate_text_embedding

        return generate_text_embedding(text, model_path)


NODE_CLASS_MAPPINGS = {
    "MLXTextEmbedding": MLXTextEmbedding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXTextEmbedding": "MLX Generate Text Embedding",
}
