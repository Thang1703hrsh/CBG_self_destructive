import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import modal_compare_common as common


class _Response:
    content = b"synthetic fixture"

    def raise_for_status(self):
        pass

    def json(self):
        return [
            {"is_safe": False, "prompt": f"prompt-{i}",
             "response": f"response-{i}", "refusal": f"refusal-{i}"}
            for i in range(6000)
        ]


class CompareSplitTests(unittest.TestCase):
    def test_seam_attack_indices_and_ctrap_disjointness(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(common.requests, "get", return_value=_Response()):
            directory = Path(temp) / "comparison"
            metadata = common.prepare_split(42, directory)
            attack = json.loads((directory / "attack.json").read_text(encoding="utf-8"))
            defense = json.loads((directory / "ctrap_defense.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["attack_count"], 1000)
            self.assertEqual(metadata["ctrap_defense_count"], 5000)
            self.assertEqual(len(set(row["prompt"] for row in attack)), 1000)
            self.assertFalse({row["prompt"] for row in attack} & {row["prompt"] for row in defense})

    def test_l40s_lora_sweep_only_changes_attack_invocations(self):
        calls = []

        def fake_run(args, cwd, env, check):
            calls.append((args, env))
            result_path = Path(args[args.index("--result_path") + 1])
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temp, patch.object(
            common.subprocess, "run", side_effect=fake_run,
        ):
            output = Path(temp)
            with redirect_stdout(io.StringIO()):
                common.evaluate_sweep(
                    "CTRAP", 42, 100, output, output / "attack.json",
                    str(output / "defense" / "model"), lambda _: None,
                    lora_attack_config="ctrap", direct_gpu_load=True,
                )

        self.assertEqual(len(calls), 12)
        self.assertTrue(all(env.get("SEAM_DIRECT_GPU_LOAD") == "1" for _, env in calls))
        self.assertEqual(sum("--lora_attack" in args for args, _ in calls), 10)
        for args, _ in calls:
            if args[args.index("--attack") + 1] == "true":
                self.assertEqual(args[args.index("--lora_attack_config") + 1], "ctrap")
            else:
                self.assertNotIn("--lora_attack", args)


if __name__ == "__main__":
    unittest.main()
