"""Overlay an epoch's replacement tensors on the exact original pose checkpoint."""
def load_adapter(model, base_path, adapter_path, expected_base_sha256):
    import hashlib
    from safetensors.torch import load_file
    h=hashlib.sha256()
    with open(base_path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    if h.hexdigest()!=expected_base_sha256:raise ValueError('Base checkpoint hash mismatch')
    model.load_state_dict(load_file(str(base_path)),strict=True)
    result=model.load_state_dict(load_file(str(adapter_path)),strict=False)
    if result.unexpected_keys:raise ValueError(result.unexpected_keys)
    return model
