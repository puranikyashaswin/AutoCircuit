import torch
from transformer_lens import HookedTransformer

_loaded_models = {}


def load_model(model_name="gpt2", device="cpu"):
    """Loads and caches a HookedTransformer. Subsequent calls return the same instance."""
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    print(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    _loaded_models[model_name] = model
    print(f"  {model.cfg.n_layers} layers, {model.cfg.n_heads} heads/layer")
    return model


def tokenize(model, text, prepend_bos=True):
    return model.to_tokens(text, prepend_bos=prepend_bos)


def forward(model, tokens):
    with torch.no_grad():
        return model(tokens)


def forward_with_cache(model, tokens):
    """Returns (logits, activation_cache). The cache holds every intermediate
    activation and is ~1.2GB for GPT-2 - we only call this once per clean
    prompt and reuse the cache across all patching iterations."""
    with torch.no_grad():
        return model.run_with_cache(tokens)


def resolve_target_token(model, target_text):
    """Converts a target string like ' Bob' to a single token ID.
    Raises if the string spans multiple tokens - IOI targets must be
    single-token names for the logit diff metric to be well-defined."""
    ids = model.to_tokens(target_text, prepend_bos=False).squeeze()
    if ids.dim() > 0 and ids.shape[0] > 1:
        raise ValueError(
            f"'{target_text}' tokenizes to {ids.shape[0]} tokens "
            f"({ids.tolist()}) - need exactly 1 for logit diff"
        )
    return int(ids.item()) if ids.dim() == 0 else int(ids[0].item())
