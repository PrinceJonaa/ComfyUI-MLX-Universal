import mlx.core as mx

from .bridge import mlx_to_torch
from .model_loader import load_embedding_model


def generate_text_embedding(text: str, model_path: str) -> tuple:
    """
    Generates text embeddings using an MLX embedding model.
    """
    model, tokenizer = load_embedding_model(model_path)

    print(f"Generating embeddings using '{model_path}'...")

    # Tokenize the input text
    # MLX embeddings models usually just take a list of strings natively or expect a standard list of ints
    # if mlx_embeddings tokenizer supports encode(), we wrap in mx.array if not natively supported,
    # but mlx_embeddings generally expects strings or lists.
    # Actually, according to the `mlx_embeddings` docs:
    # "input_ids = tokenizer.encode(text, return_tensors='mlx')" is explicitly supported by its utilities!
    # Wait, the code reviewer noted: "Hugging Face tokenizers generally do not support 'mlx' as a valid argument for return_tensors."
    # Let's fix it by passing the raw string or using mx.array. We will use the approach standard for mlx_lm/mlx_embeddings.

    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "pad_token_id"):
        # standard HF tokenizer behavior fallback
        try:
            input_ids = tokenizer.encode(text, return_tensors="mlx")
        except (TypeError, ValueError):
            # fallback if 'mlx' is not supported
            input_ids_list = tokenizer.encode(text)
            input_ids = mx.array([input_ids_list])
    else:
        # Just in case, mlx_embeddings actually patched encode to support "mlx" as per its docs
        input_ids = tokenizer.encode(text, return_tensors="mlx")

    # Generate embeddings
    outputs = model(input_ids)

    # Extract the mean pooled and normalized embeddings
    text_embeds = outputs.text_embeds

    # Explicitly evaluate the lazy array before bridging to PyTorch
    mx.eval(text_embeds)
    print("Embedding generation complete.")

    # Route through the tensor bridge
    text_embeds_torch = mlx_to_torch(text_embeds)

    return (text_embeds_torch,)
