from torch import nn
from .clip_model import CLIP
from .clip_surgery_model import CLIPSurgery


def convert_weights(model: nn.Module):
    """
    Convert applicable model parameters to fp16.

    Notes
    -----
    - Half precision is commonly used for CLIP inference to save memory and
      accelerate compute while preserving quality.
    - This conversion mirrors practices from OpenAI CLIP.
    - Keep in fp32 for SAM or CPU-only scenarios if numerical stability is a
      concern.
    """

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(name: str, state_dict: dict):
    """
    Build a CLIP or CLIP-Surgery model from a checkpoint state dict.

    Parameters
    ----------
    name : str
        Model identifier. If it contains the prefix "CS-", the modified
        CLIP-Surgery architecture is instantiated.
    state_dict : dict
        Weights loaded from the original CLIP checkpoints.

    Returns
    -------
    torch.nn.Module (eval mode)

    How it relates to papers
    ------------------------
    - The vanilla CLIP graph structure follows OpenAI CLIP.
    - The CLIP-Surgery path modifies the attention in the last ViT blocks
      to emphasize value-only pathways and token-level spatial outputs, as in
      the explainability direction discussed in "A Closer Look at the
      Explainability of CLIP" and used for generating token maps that guide
      SAM points in AlignSAM.
    """
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    # Instantiate CLIP-Surgery when explicitly requested by name
    if 'CS-' in name:
        model = CLIPSurgery(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
    else:
        model = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # Keeping fp32 by default for numerical stability with downstream tasks
    # (e.g., SAM point maps). Enable if you prefer fp16 memory/latency.
    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
