def _map_vae_keys(k):
    if "mid_blocks.block_1" in k:
        k = k.replace("mid_blocks.block_1", "mid_blocks.0")
    if "mid_blocks.block_2" in k:
        k = k.replace("mid_blocks.block_2", "mid_blocks.2")
    if "mid_blocks.attn_1" in k:
        k = k.replace("mid_blocks.attn_1", "mid_blocks.1")
    if "mid" in k:
        k = k.replace("mid", "mid_blocks")
    if ".norm." in k:
        k = k.replace(".norm.", ".group_norm.")
    if "norm_out" in k:
        k = k.replace("norm_out", "conv_norm_out")
    if ".q" in k:
        k = k.replace(".q", ".query_proj")
    if ".k" in k:
        k = k.replace(".k", ".key_proj")
    if ".v" in k:
        k = k.replace(".v", ".value_proj")
    if ".proj_out" in k:
        k = k.replace(".proj_out", ".out_proj")
    if ".block." in k:
        k = k.replace(".block.", ".resnets.")
    if ".nin_shortcut." in k:
        k = k.replace(".nin_shortcut.", ".conv_shortcut.")
    return k


def _reshape_vae_weights(k, v):
    if "resnets" in k and "conv" in k and "weight" in k:
        return v.transpose(0, 2, 3, 1)
    if "mid_blocks" in k and "conv" in k and "weight" in k:
        return v.transpose(0, 2, 3, 1)
    if "conv_shortcut.weight" in k:
        return v[:, 0, 0, :]
    if "proj.weight" in k:
        return v[:, :, 0, 0]
    return v


def _common_vae_adjustments(state_dict):
    # Filter out MMDIT related tensors
    state_dict = {k: v for k, v in state_dict.items() if "diffusion_model." not in k}

    state_dict = {_map_vae_keys(k): v for k, v in state_dict.items()}

    # reshape weights
    state_dict = {k: _reshape_vae_weights(k, v) for k, v in state_dict.items()}

    if "conv_in.weight" in state_dict:
        state_dict["conv_in.weight"] = state_dict["conv_in.weight"].transpose(
            0, 2, 3, 1
        )
    if "conv_out.weight" in state_dict:
        state_dict["conv_out.weight"] = state_dict["conv_out.weight"].transpose(
            0, 2, 3, 1
        )

    return state_dict


def vae_decoder_state_dict_adjustments(state_dict, prefix="decoder."):
    # Keep only the keys that have the prefix
    state_dict = {k: v for k, v in state_dict.items() if prefix in k}
    state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items()}

    state_dict = {k.replace("up.", "up_blocks."): v for k, v in state_dict.items()}
    state_dict = {
        k.replace(".upsample.conv.", ".upsample."): v for k, v in state_dict.items()
    }

    state_dict = _common_vae_adjustments(state_dict)

    # reshape weights
    state_dict = {
        k: v.transpose(0, 2, 3, 1) if "upsample" in k and "weight" in k else v
        for k, v in state_dict.items()
    }

    return state_dict


def vae_encoder_state_dict_adjustments(state_dict, prefix="encoder."):
    # Keep only the keys that have the prefix
    state_dict = {k: v for k, v in state_dict.items() if prefix in k}
    state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items()}

    state_dict = {k.replace("down.", "down_blocks."): v for k, v in state_dict.items()}
    state_dict = {
        k.replace(".downsample.conv.", ".downsample."): v for k, v in state_dict.items()
    }

    state_dict = _common_vae_adjustments(state_dict)

    # reshape weights
    state_dict = {
        k: v.transpose(0, 2, 3, 1) if "downsample" in k and "weight" in k else v
        for k, v in state_dict.items()
    }

    return state_dict
