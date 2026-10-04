import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import hf_checkpoint_push as push


class FakeApi:
    def __init__(self, files=(), private=True):
        self.files = set(files)
        self.private = private
        self.uploads = []

    def create_repo(self, **kwargs):
        self.created = kwargs

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=self.private)

    def list_repo_files(self, **kwargs):
        return sorted(self.files)

    def upload_folder(self, *, folder_path, path_in_repo=None, **kwargs):
        prefix = f"{path_in_repo}/" if path_in_repo else ""
        self.files.update(prefix + path.name for path in Path(folder_path).iterdir() if path.is_file())
        self.uploads.append(("folder", path_in_repo))

    def upload_file(self, *, path_in_repo, **kwargs):
        self.files.add(path_in_repo)
        self.uploads.append(("file", path_in_repo))


class HfCheckpointPushTests(unittest.TestCase):
    def test_preflight_probes_write_and_refuses_existing_model(self):
        api = FakeApi()
        with patch.object(push, "_api", return_value=api):
            self.assertEqual(
                push.ensure_private_empty_repo("owner/new-model"),
                "https://huggingface.co/owner/new-model",
            )
        self.assertEqual(api.created["private"], True)
        self.assertIn(("file", ".train-push-preflight"), api.uploads)

        api = FakeApi(files={"config.json"})
        with patch.object(push, "_api", return_value=api):
            with self.assertRaises(FileExistsError):
                push.ensure_private_empty_repo("owner/existing-model")
        self.assertFalse(api.uploads)

        with patch.object(push, "_api", return_value=FakeApi(files={".train-push-preflight"})):
            with self.assertRaises(FileExistsError):
                push.ensure_private_empty_repo("owner/already-reserved")

    def test_upload_requires_500_steps_and_checks_remote_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            model = output / "defense" / "model"
            adapter = output / "defense" / "adapter"
            model.mkdir(parents=True)
            adapter.mkdir()
            for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
                (model / filename).write_text("test", encoding="utf-8")
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output / "manifest.json").write_text("{}", encoding="utf-8")
            metrics = output / "defense" / "train_metrics.json"
            metrics.write_text(json.dumps({"global_step": 499}), encoding="utf-8")

            api = FakeApi(private=True)
            with patch.object(push, "_api", return_value=api):
                with self.assertRaises(ValueError):
                    push.push_checkpoint("owner/model", output, adapter=True)
                self.assertFalse(api.uploads)
                metrics.write_text(json.dumps({"global_step": 500}), encoding="utf-8")
                url = push.push_checkpoint("owner/model", output, adapter=True)
            self.assertEqual(url, "https://huggingface.co/owner/model")
            self.assertIn("adapter/adapter_config.json", api.files)
            self.assertIn("train_metrics.json", api.files)

    def test_upload_accepts_requested_10000_steps_but_not_500(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            model = output / "defense" / "model"
            adapter = output / "defense" / "adapter"
            model.mkdir(parents=True)
            adapter.mkdir()
            for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
                (model / filename).write_text("test", encoding="utf-8")
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output / "manifest.json").write_text("{}", encoding="utf-8")
            metrics = output / "defense" / "train_metrics.json"
            metrics.write_text(json.dumps({"global_step": 500}), encoding="utf-8")

            api = FakeApi(private=True)
            with patch.object(push, "_api", return_value=api):
                with self.assertRaises(ValueError):
                    push.push_checkpoint(
                        "owner/model", output, adapter=True, expected_steps=10000,
                    )
                self.assertFalse(api.uploads)
                metrics.write_text(json.dumps({"global_step": 10000}), encoding="utf-8")
                push.push_checkpoint(
                    "owner/model", output, adapter=True, expected_steps=10000,
                )
            self.assertIn("adapter/adapter_config.json", api.files)


if __name__ == "__main__":
    unittest.main()
