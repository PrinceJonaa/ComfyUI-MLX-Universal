import mlx.core as mx


def _map_mmdit_keys(k):
    if "y_embedder.mlp" in k:
        k = k.replace("y_embedder.mlp", "y_embedder.mlp.layers")
    if "t_embedder.mlp" in k:
        k = k.replace("t_embedder.mlp", "t_embedder.mlp.layers")
    if "adaLN_modulation" in k:
        k = k.replace("adaLN_modulation", "adaLN_modulation.layers")
    if "al_layer" in k:
        k = k.replace("al_layer", "final_layer")
    if "joint_blocks" in k:
        k = k.replace("joint_blocks", "multimodal_transformer_blocks")
    if "context_block" in k:
        k = k.replace("context_block", "text_transformer_block")
    if "x_block" in k:
        k = k.replace("x_block", "image_transformer_block")
    return k


def _split_qkv_mmdit(state_dict):
    keys_to_pop = []
    state_dict_update = {}
    for k in state_dict:
        if "attn.qkv" in k:
            keys_to_pop.append(k)
            for name, weight in zip(["q", "k", "v"], mx.split(state_dict[k], 3)):
                state_dict_update[k.replace("attn.qkv", f"attn.{name}_proj")] = (
                    weight if "weight" in k else weight
                )

    for k in keys_to_pop:
        state_dict.pop(k)
    state_dict.update(state_dict_update)
    return state_dict


def mmdit_state_dict_adjustments(state_dict, prefix=""):
    # Remove prefix
    state_dict = {k.lstrip(prefix): v for k, v in state_dict.items()}

    state_dict = {_map_mmdit_keys(k): v for k, v in state_dict.items()}

    state_dict = _split_qkv_mmdit(state_dict)

    state_dict = {
        k.replace("attn.proj", "attn.o_proj"): (
            v if "attn.proj" in k and "weight" in k else v
        )
        for k, v in state_dict.items()
    }

    # Filter out VAE Decoder related tensors
    state_dict = {k: v for k, v in state_dict.items() if "decoder." not in k}

    # Filter out k_proj.bias related tensors
    state_dict = {k: v for k, v in state_dict.items() if "k_proj.bias" not in k}

    # Filter out teacher_model related tensors
    state_dict = {k: v for k, v in state_dict.items() if "teacher_model." not in k}

    # Remap pos_embed buffer -> nn.Embedding
    state_dict = {
        k.replace("pos_embed", "x_pos_embedder.pos_embed.weight"): (
            v[0] if "pos_embed" in k else v
        )
        for k, v in state_dict.items()
    }

    # Transpose x_embedder.proj.weight
    if "x_embedder.proj.weight" in state_dict:
        state_dict["x_embedder.proj.weight"] = state_dict[
            "x_embedder.proj.weight"
        ].transpose(0, 2, 3, 1)

    return state_dict
