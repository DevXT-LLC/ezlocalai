"""Exercise real Accelerate loaders without downloading model weights."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import accelerate
import accelerate.utils.modeling as modeling
import torch

from ezlocalai.checkpoint_security import harden_accelerate_checkpoint_loading


class CheckpointSecurityTests(unittest.TestCase):
    def setUp(self):
        harden_accelerate_checkpoint_loading()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshots" / "revision"
        self.snapshot.mkdir(parents=True)
        self.index = self.snapshot / "pytorch_model.bin.index.json"
        self.model = torch.nn.Linear(2, 2)

    def write_index(self, name, legacy=False):
        weights = {key: name for key in self.model.state_dict()}
        self.index.write_text(
            json.dumps(weights if legacy else {"weight_map": weights})
        )

    def assert_rejected(self, checkpoint):
        for loader in (
            modeling.load_checkpoint_in_model,
            accelerate.load_checkpoint_in_model,
            accelerate.load_checkpoint_and_dispatch,
        ):
            with self.subTest(loader=loader.__name__):
                with patch.object(modeling, "load_state_dict") as load:
                    with self.assertRaises(ValueError):
                        loader(self.model, checkpoint, device_map={"": "cpu"})
                    load.assert_not_called()

    def test_escaping_shards_are_rejected_before_loading(self):
        outside = self.root / "outside.bin"
        torch.save(self.model.state_dict(), outside)
        for name in ("../../outside.bin", str(outside), "nested/../../../outside.bin"):
            for legacy in (False, True):
                with self.subTest(name=name, legacy=legacy):
                    self.write_index(name, legacy=legacy)
                    self.assert_rejected(self.index)
                    self.assert_rejected(self.snapshot)

    def test_nonregular_shards_and_indexes_are_rejected(self):
        fifo = self.snapshot / "pipe.bin"
        os.mkfifo(fifo)
        self.write_index(fifo.name)
        self.assert_rejected(self.snapshot)
        self.write_index(".")
        self.assert_rejected(self.index)
        self.index.unlink()
        os.mkfifo(self.index)
        self.assert_rejected(self.snapshot)
        self.assert_rejected(self.index)

    def test_monolithic_fifo_is_rejected(self):
        os.mkfifo(self.snapshot / "pytorch_model.bin")
        self.assert_rejected(self.snapshot)

    def test_nested_shards_and_hub_cache_symlinks_load(self):
        blobs = self.root / "blobs"
        blobs.mkdir()
        blob = blobs / "weights"
        torch.save(self.model.state_dict(), blob)
        nested = self.snapshot / "nested"
        nested.mkdir()
        (nested / "shard.bin").symlink_to(blob)
        for legacy in (False, True):
            self.write_index("./nested/shard.bin", legacy=legacy)
            for checkpoint in (self.index, self.snapshot):
                loaded = torch.nn.Linear(2, 2)
                accelerate.load_checkpoint_and_dispatch(
                    loaded, checkpoint, device_map={"": "cpu"}
                )
                for key, expected in self.model.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], expected)

    def test_monolithic_checkpoint_keeps_priority_over_unused_index(self):
        self.write_index("../../outside.bin")
        torch.save(self.model.state_dict(), self.snapshot / "pytorch_model.bin")
        loaded = torch.nn.Linear(2, 2)
        accelerate.load_checkpoint_in_model(
            loaded, self.snapshot, device_map={"": "cpu"}
        )
        torch.testing.assert_close(loaded.weight, self.model.weight)

    def test_patch_is_idempotent(self):
        original = modeling.load_checkpoint_in_model
        harden_accelerate_checkpoint_loading()
        self.assertIs(modeling.load_checkpoint_in_model, original)


if __name__ == "__main__":
    unittest.main()
