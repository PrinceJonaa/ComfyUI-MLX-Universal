import mlx.core as mx


def _map_blocks(k):
    if "double_blocks" in k:
        k = k.replace("double_blocks", "multimodal_transformer_blocks")
    if "single_blocks" in k:
        k = k.replace("single_blocks", "unified_transformer_blocks")
    if "txt_attn" in k:
        k = k.replace("txt_attn", "text_transformer_block.attn")
    if "img_attn" in k:
        k = k.replace("img_attn", "image_transformer_block.attn")
    if "txt_mlp.0" in k:
        k = k.replace("txt_mlp.0", "text_transformer_block.mlp.fc1")
    if "txt_mlp.2" in k:
        k = k.replace("txt_mlp.2", "text_transformer_block.mlp.fc2")
    if "img_mlp.0" in k:
        k = k.replace("img_mlp.0", "image_transformer_block.mlp.fc1")
    if "img_mlp.2" in k:
        k = k.replace("img_mlp.2", "image_transformer_block.mlp.fc2")
    if "img_mod.lin" in k:
        k = k.replace(
            "img_mod.lin", "image_transformer_block.adaLN_modulation.layers.1"
        )
    if "txt_mod.lin" in k:
        k = k.replace("txt_mod.lin", "text_transformer_block.adaLN_modulation.layers.1")
    if ".proj" in k:
        k = k.replace(".proj", ".o_proj")
    if ".attn.norm.key_norm.scale" in k:
        k = k.replace(".attn.norm.key_norm.scale", ".qk_norm.k_norm.weight")
    if ".attn.norm.query_norm.scale" in k:
        k = k.replace(".attn.norm.query_norm.scale", ".qk_norm.q_norm.weight")
    if ".modulation.lin" in k:
        k = k.replace(".modulation.lin", ".transformer_block.adaLN_modulation.layers.1")
    if ".norm.key_norm.scale" in k:
        k = k.replace(
            ".norm.key_norm.scale", ".transformer_block.qk_norm.k_norm.weight"
        )
    if ".norm.query_norm.scale" in k:
        k = k.replace(
            ".norm.query_norm.scale", ".transformer_block.qk_norm.q_norm.weight"
        )
    return k


def _map_embedders(k):
    if "img_in." in k:
        k = k.replace("img_in.", "x_embedder.proj.")
    if "txt_in." in k:
        k = k.replace("txt_in.", "context_embedder.")
    if "time_in." in k:
        k = k.replace("time_in.", "t_embedder.")
    if "vector_in." in k:
        k = k.replace("vector_in.", "y_embedder.")
    if ".in_layer." in k:
        k = k.replace(".in_layer.", ".mlp.layers.0.")
    if ".out_layer." in k:
        k = k.replace(".out_layer.", ".mlp.layers.2.")
    if "final_layer.adaLN_modulation.1" in k:
        k = k.replace(
            "final_layer.adaLN_modulation.1", "final_layer.adaLN_modulation.layers.1"
        )
    return k


def _split_qkv(state_dict):
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


def _split_linear1(state_dict, hidden_size):
    keys_to_pop = []
    state_dict_update = {}
    for k in state_dict:
        if ".linear1" in k:
            keys_to_pop.append(k)
            for name, weight in zip(
                ["attn.q", "attn.k", "attn.v", "mlp.fc1"],
                mx.split(
                    state_dict[k],
                    [
                        hidden_size,
                        2 * hidden_size,
                        3 * hidden_size,
                    ],
                ),
            ):
                if name == "mlp.fc1":
                    state_dict_update[
                        k.replace(".linear1", f".transformer_block.{name}")
                    ] = weight if "weight" in k else weight
                else:
                    state_dict_update[
                        k.replace(".linear1", f".transformer_block.{name}_proj")
                    ] = weight if "weight" in k else weight

    for k in keys_to_pop:
        state_dict.pop(k)
    state_dict.update(state_dict_update)
    return state_dict


def _split_linear2(state_dict, hidden_size):
    keys_to_pop = []
    state_dict_update = {}
    for k in state_dict:
        if ".linear2" in k:
            keys_to_pop.append(k)
            if "bias" in k:
                state_dict_update[
                    k.replace(".linear2", ".transformer_block.attn.o_proj")
                ] = state_dict[k]
                state_dict_update[
                    k.replace(".linear2", ".transformer_block.mlp.fc2")
                ] = state_dict[k]
            else:
                for name, weight in zip(
                    ["attn.o", "mlp.fc2"],
                    mx.split(
                        state_dict[k],
                        [hidden_size],
                        axis=1,
                    ),
                ):
                    if name == "mlp.fc2":
                        state_dict_update[
                            k.replace(".linear2", f".transformer_block.{name}")
                        ] = weight if "weight" in k else weight
                    else:
                        state_dict_update[
                            k.replace(".linear2", f".transformer_block.{name}_proj")
                        ] = weight if "weight" in k else weight

    for k in keys_to_pop:
        state_dict.pop(k)
    state_dict.update(state_dict_update)
    return state_dict


def flux_state_dict_adjustments(state_dict, prefix="", hidden_size=3072, mlp_ratio=4):
    state_dict = {_map_blocks(k): v for k, v in state_dict.items()}

    state_dict = _split_qkv(state_dict)

    state_dict = _split_linear1(state_dict, hidden_size)

    state_dict = _split_linear2(state_dict, hidden_size)

    state_dict = {_map_embedders(k): v for k, v in state_dict.items()}

    state_dict["x_embedder.proj.weight"] = mx.expand_dims(
        mx.expand_dims(state_dict["x_embedder.proj.weight"], axis=1), axis=1
    )

    return state_dict
