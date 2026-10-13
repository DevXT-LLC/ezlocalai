"""Guard Accelerate checkpoint indexes until its upstream path fix is released.

Accelerate 1.15.0 still trusts shard names from downloaded JSON indexes
(GHSA-4j2p-28q2-5m79). Keep this shim until a patched release is verified.
"""

from functools import wraps
import json
import os
import threading


_patch_lock = threading.Lock()


def _regular_file(path):
    # Follow regular-file symlinks: Hub snapshots link to a shared blobs folder.
    # Reject FIFOs/devices before a loader can open them and block indefinitely.
    if not os.path.isfile(path):
        raise ValueError("Checkpoint artifacts must be regular files")


def validate_checkpoint(checkpoint):
    """Validate the files Accelerate will select, preserving Hub cache layouts."""
    checkpoint = os.path.abspath(os.fspath(checkpoint))
    if os.path.isdir(checkpoint):
        names = os.listdir(checkpoint)
        for name in ("pytorch_model.bin", "model.safetensors"):
            if name in names:
                _regular_file(os.path.join(checkpoint, name))
                return
        indexes = [name for name in names if name.endswith(".index.json")]
        if len(indexes) != 1:
            # Let Accelerate report its usual missing/ambiguous-index error.
            return
        checkpoint = os.path.join(checkpoint, indexes[0])

    _regular_file(checkpoint)
    if not checkpoint.endswith(".json"):
        return
    with open(checkpoint, encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, dict):
        raise ValueError("Checkpoint index must be an object")
    weight_map = index.get("weight_map", index)
    if not isinstance(weight_map, dict):
        raise ValueError("Checkpoint weight map must be an object")
    folder = os.path.dirname(checkpoint)
    for shard in weight_map.values():
        if not isinstance(shard, str) or not shard or os.path.isabs(shard):
            raise ValueError("Checkpoint shards must use relative paths")
        path = os.path.abspath(os.path.join(folder, shard))
        if os.path.commonpath((folder, path)) != folder:
            raise ValueError("Checkpoint shard escapes its directory")
        _regular_file(path)


def harden_accelerate_checkpoint_loading():
    """Install before importing model integrations that capture loader aliases."""
    try:
        import accelerate
    except ModuleNotFoundError as exc:
        if exc.name == "accelerate":
            return  # Optional on installations without Python model loaders.
        raise
    import accelerate.big_modeling as big_modeling
    import accelerate.utils as utils
    import accelerate.utils.modeling as modeling

    with _patch_lock:
        original = modeling.load_checkpoint_in_model
        if getattr(original, "_ezlocalai_checkpoint_guard", False):
            return

        @wraps(original)
        def guarded(model, checkpoint, *args, **kwargs):
            validate_checkpoint(checkpoint)
            return original(model, checkpoint, *args, **kwargs)

        guarded._ezlocalai_checkpoint_guard = True
        # load_checkpoint_and_dispatch uses big_modeling's imported alias.
        for module in (accelerate, utils, modeling, big_modeling):
            if getattr(module, "load_checkpoint_in_model", None) is original:
                module.load_checkpoint_in_model = guarded
