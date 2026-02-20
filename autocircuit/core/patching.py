import torch
from transformer_lens.hook_points import HookPoint

from autocircuit.core.model_utils import forward_with_cache


# TransformerLens hook point names - using hook_z (pre-W_O head output) rather
# than hook_result because TL doesn't expose hook_result. hook_z gives us the
# (batch, pos, n_heads, d_head) tensor we need for per-head patching.
def _attn_hook_name(layer):
    return f"blocks.{layer}.attn.hook_z"

def _mlp_hook_name(layer):
    return f"blocks.{layer}.hook_mlp_out"


def _replace_head_hook(cached_activations, layer, head):
    """Returns a hook that overwrites one head's z-output with the cached version."""
    key = _attn_hook_name(layer)
    def hook(act, hook=None):
        act[:, :, head, :] = cached_activations[key][:, :, head, :]
        return act
    return hook


def _replace_mlp_hook(cached_activations, layer):
    key = _mlp_hook_name(layer)
    def hook(act, hook=None):
        act[:] = cached_activations[key]
        return act
    return hook


def cache_clean_run(model, clean_tokens):
    return forward_with_cache(model, clean_tokens)


def corrupted_baseline(model, corrupted_tokens):
    with torch.no_grad():
        return model(corrupted_tokens)


def patch_head(model, corrupted_tokens, clean_cache, layer, head):
    """Runs corrupted prompt but swaps in one attention head's clean activation."""
    name = _attn_hook_name(layer)
    hook = _replace_head_hook(clean_cache, layer, head)
    with torch.no_grad():
        return model.run_with_hooks(corrupted_tokens, fwd_hooks=[(name, hook)])


def patch_mlp(model, corrupted_tokens, clean_cache, layer):
    name = _mlp_hook_name(layer)
    hook = _replace_mlp_hook(clean_cache, layer)
    with torch.no_grad():
        return model.run_with_hooks(corrupted_tokens, fwd_hooks=[(name, hook)])


def patch_components(model, corrupted_tokens, clean_cache, components):
    """Patches multiple components in a single forward pass.
    Each component is a dict with 'type', 'layer', and optionally 'head'."""
    hooks = []
    for c in components:
        if c["type"] == "attn_head":
            hooks.append((_attn_hook_name(c["layer"]),
                          _replace_head_hook(clean_cache, c["layer"], c["head"])))
        elif c["type"] == "mlp":
            hooks.append((_mlp_hook_name(c["layer"]),
                          _replace_mlp_hook(clean_cache, c["layer"])))
        else:
            raise ValueError(f"bad component type: {c['type']}")

    with torch.no_grad():
        return model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)
