"""Wake-word storage boundaries and compatibility of restricted checkpoints."""

import ast
import asyncio
import io
import json
import logging
import pickle
from pathlib import Path
import tempfile
from typing import Optional
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.testclient import TestClient
from pydantic import BaseModel

from ezlocalai.WakeWord import ModelConfig, WakeWordManager, WakeWordTrainer
from ezlocalai.WakeWordPaths import contained_path, wakeword_directory


def mark_unpickled(path):
    Path(path).touch()


class UntrustedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return mark_unpickled, (str(self.marker),)


class WakeWordSecurityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manager = WakeWordManager(
            models_dir=self.root / "models",
            samples_dir=self.root / "samples",
            jobs_dir=self.root / "jobs",
            voices_dir=self.root / "voices",
        )

    def model(self, word="Hey Computer"):
        folder = wakeword_directory(self.manager.models_dir, word)
        folder.mkdir()
        (folder / "model.pt").write_bytes(b"fixture")
        (folder / "config.json").write_text(json.dumps({"word": word}))
        return folder

    def test_normal_names_keep_existing_directories_and_delete_only_target(self):
        for word, name in (
            ("Hey Computer", "hey_computer"),
            ("Héllo-XT!", "héllo-xt!"),
        ):
            folder = self.model(word)
            self.assertEqual(folder.name, name)
            self.assertEqual(self.manager.get_model_for_word(word), folder)
            self.assertTrue(self.manager.delete_model(word))
            self.assertFalse(folder.exists())
        self.assertEqual(self.manager.list_available_models(), [])

    def test_traversal_and_root_names_cannot_read_delete_or_start_training(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "model.pt").write_bytes(b"preserve")
        for word in (
            "",
            ".",
            "..",
            "../outside",
            str(outside),
            "a/../../outside",
            "..\\outside",
            "bad\x00word",
        ):
            with self.subTest(word=word):
                for operation in (
                    self.manager.get_model_for_word,
                    self.manager.delete_model,
                ):
                    with self.assertRaises(ValueError):
                        operation(word)
                with self.assertRaises(ValueError):
                    asyncio.run(self.manager.create_job(word))
        self.assertEqual((outside / "model.pt").read_bytes(), b"preserve")
        self.assertEqual(self.manager.jobs, {})
        self.assertEqual(list(self.manager.samples_dir.iterdir()), [])

    def test_sibling_prefix_and_symlink_escapes_are_rejected(self):
        outside = self.root / "models-private"
        outside.mkdir()
        (outside / "model.pt").write_bytes(b"private")
        (self.manager.models_dir / "linked").symlink_to(
            outside, target_is_directory=True
        )
        with self.assertRaises(ValueError):
            self.manager.get_model_for_word("linked")
        self.assertEqual(self.manager.list_available_models(), [])
        for filename in ("model.pt", "config.json", "model.onnx", "model.espdl"):
            with self.subTest(filename=filename):
                model = self.model("safe")
                target = model / filename
                target.unlink(missing_ok=True)
                target.symlink_to(outside / "model.pt")
                with self.assertRaises(ValueError):
                    self.manager.get_model_for_word("safe")
                self.assertEqual(self.manager.list_available_models(), [])
                target.unlink()
                import shutil

                shutil.rmtree(model)
        self.assertEqual((outside / "model.pt").read_bytes(), b"private")

    def test_existing_checkpoint_roundtrip_and_inference(self):
        trainer = WakeWordTrainer(ModelConfig(word="Hey Computer"), device="cpu")
        folder = self.manager.models_dir / "hey_computer"
        with mock.patch.object(
            torch.onnx,
            "export",
            side_effect=RuntimeError("export not needed for checkpoint test"),
        ):
            trainer.save(folder)
        loaded = WakeWordTrainer.load(folder, device="cpu")
        self.assertEqual(loaded.config, trainer.config)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(loaded.model.state_dict()[name], value)
        audio = io.BytesIO()
        sf.write(audio, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
        before = trainer.predict(audio.getvalue())
        after = loaded.predict(audio.getvalue())
        self.assertEqual(before, after)

    def test_checkpoint_cannot_execute_pickle_globals(self):
        folder = self.model("safe")
        marker = self.root / "unpickled"
        torch.save(
            {"model_state_dict": UntrustedCheckpoint(marker)}, folder / "model.pt"
        )
        with self.assertRaises(pickle.UnpicklingError):
            WakeWordTrainer.load(folder, device="cpu")
        self.assertFalse(marker.exists())

    def client(self):
        # Register the actual route functions without initializing unrelated GPU
        # models. Exercise FastAPI validation, response models and FileResponse.
        tree = ast.parse(Path(__file__).with_name("app.py").read_text())
        names = {
            "validated_wakeword",
            "get_wakeword_model",
            "get_wakeword_model_config",
            "delete_wakeword_model",
            "train_wakeword_model",
            "predict_wakeword",
            "predict_wakeword_file",
        }
        nodes = [
            n
            for n in tree.body
            if getattr(n, "name", "") in names
            or (isinstance(n, ast.ClassDef) and n.name.startswith("WakeWord"))
        ]
        app = FastAPI()
        namespace = {
            "__name__": __name__,
            "app": app,
            "Depends": Depends,
            "HTTPException": HTTPException,
            "BaseModel": BaseModel,
            "Optional": Optional,
            "verify_api_key": lambda: "test",
            "is_wakeword_enabled": lambda: True,
            "contained_path": contained_path,
            "logging": logging,
            "UploadFile": UploadFile,
            "File": File,
        }
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"),
            namespace,
        )
        patcher = mock.patch(
            "ezlocalai.WakeWord.get_wakeword_manager", return_value=self.manager
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return TestClient(app)

    def test_endpoint_download_config_missing_and_invalid_training(self):
        folder = self.model()
        with self.client() as client:
            response = client.get("/v1/wakeword/models/Hey%20Computer")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b"fixture")
            self.assertIn(
                "hey_computer_model.pt", response.headers["content-disposition"]
            )
            self.assertEqual(
                client.get("/v1/wakeword/models/Hey%20Computer/config").json(),
                {"word": "Hey Computer"},
            )
            self.assertEqual(client.get("/v1/wakeword/models/missing").status_code, 404)
            self.assertEqual(
                client.get(
                    "/v1/wakeword/models/Hey%20Computer?format=onnx"
                ).status_code,
                404,
            )
            self.assertEqual(
                client.post(
                    "/v1/wakeword/train", json={"word": "../outside"}
                ).status_code,
                400,
            )
            self.assertEqual(
                client.post(
                    "/v1/wakeword/predict/..%5Coutside", json={"audio_base64": ""}
                ).status_code,
                400,
            )
            self.assertEqual(
                client.delete("/v1/wakeword/models/Hey%20Computer").status_code, 200
            )
            self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
