"""Hub upload of a finished run (src/experiments/hf_upload.py), against a fake HfApi."""
import os
os.environ.setdefault("USE_TF", "0")
import json
from pathlib import Path
import tempfile
import unittest

import torch
import yaml
from safetensors.torch import load_file

from src.experiments.hf_upload import check_repo_id, upload_run
from test_cbg_train_v2 import tiny_model


class FakeApi:
    def __init__(self):
        self.repos, self.commits = [], []

    def create_repo(self, repo_id, **kwargs):
        self.repos.append((repo_id, kwargs))
        return f"https://huggingface.co/{repo_id}"

    def create_commit(self, repo_id, operations, **kwargs):
        # Files must still exist while the commit runs (staging is cleaned up afterwards).
        self.commits.append({op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in operations})
        return type("Info", (), dict(oid="abc123", commit_url="https://huggingface.co/x/commit/abc123"))()


class HfUploadTest(unittest.TestCase):
    def make_run(self, root):
        run = root / "run" / "train"
        tiny_model(run / "final")
        (run / "config.yaml").write_text(yaml.safe_dump(dict(model="meta-llama/Llama-2-7b-chat-hf",
                                                             method=dict(beta=.01, lazy_period=4, lazy_mode="reuse"))))
        (run / "metrics.json").write_text(json.dumps(dict(status="complete", steps=500)))
        (run / "train_log.jsonl").write_text(json.dumps({"step": 0, "geo/log_eta_k": -3.}) + "\n" +
                                             json.dumps({"step": 500, "geo/log_eta_k": 1.5}) + "\n")
        return run

    def test_uploads_model_records_and_card_in_one_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            api = FakeApi()
            record = upload_run(run, "user/cbg-7b", api=api, base_revision="f5db02d")
            self.assertEqual(api.repos, [("user/cbg-7b", dict(repo_type="model", private=True, exist_ok=True))])
            files = api.commits[0]
            self.assertIn("model.safetensors", files)
            self.assertIn("config.json", files)
            self.assertIn("training/train_log.jsonl", files)
            card = files["README.md"].decode()
            self.assertIn("base_model: meta-llama/Llama-2-7b-chat-hf", card)
            self.assertIn("delta log eta_k | 4.5", card)
            self.assertIn("f5db02d", card)
            self.assertEqual(files["model.safetensors"], (run / "final" / "model.safetensors").read_bytes())
            self.assertEqual(json.loads((run / "hf_upload.json").read_text())["commit"], record["commit"])

    def test_bfloat16_cast_and_input_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            api = FakeApi()
            upload_run(run, "user/cbg-7b-bf16", api=api, dtype="bfloat16", private=False)
            self.assertFalse(api.repos[0][1]["private"])
            staged = Path(tmp) / "staged.safetensors"
            staged.write_bytes(api.commits[0]["model.safetensors"])
            original = load_file(run / "final" / "model.safetensors")
            for name, value in load_file(staged).items():
                self.assertEqual(value.dtype, torch.bfloat16)
                torch.testing.assert_close(value, original[name].to(torch.bfloat16), rtol=0, atol=0)
            self.assertEqual(json.loads(api.commits[0]["config.json"])["torch_dtype"], "bfloat16")
            with self.assertRaises(ValueError):
                check_repo_id("no-namespace")
            (run / "metrics.json").write_text(json.dumps(dict(status="running")))
            with self.assertRaises(FileNotFoundError):
                upload_run(run, "user/x", api=api)


if __name__ == "__main__":
    unittest.main()
