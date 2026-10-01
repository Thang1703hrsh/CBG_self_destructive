from __future__ import annotations

import unittest

from src.experiments.length_audit import audit_lengths


class CharacterTokenizer:
    bos_token_id = 1

    def encode(self, text, *, add_special_tokens=False):
        return [ord(character) for character in text]


class LengthAuditTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.safety = [dict(prompt="short", response="harm", refusal="no") for _ in range(20)]
        self.benign = [dict(instruction="short", input="", output="okay") for _ in range(20)]

    def test_exact_five_percent_fails_and_pair_shared_prompt_is_counted(self):
        # The long harmful response forces a shorter shared prompt for *both*
        # branches. Counting only the short refusal's isolated length misses it.
        self.safety[0]["response"] = "x" * 220
        report = audit_lengths(self.safety, self.benign, range(20), range(20), self.tokenizer,
                               safety_max_length=128, benign_max_length=128)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["safety"]["affected_pairs"], 1)
        self.assertEqual(report["safety"]["affected_refusal"], 1)
        self.assertEqual(report["safety"]["prompt_clipped_pairs"], 1)
        self.assertEqual(report["safety"]["harmful_response_clipped"], 1)
        self.assertEqual(report["safety"]["affected_fraction"], 0.05)
        self.assertEqual(report["benign"]["affected"], 0)

    def test_full_lengths_include_prompt_response_and_bos(self):
        report = audit_lengths(self.safety, self.benign, range(20), range(20), self.tokenizer,
                               safety_max_length=128, benign_max_length=128)
        self.assertEqual(report["status"], "passed")
        expected = 1 + len("Question: short\nAnswer:\n") + len("harm")
        self.assertEqual(report["safety"]["harmful_full_lengths"]["min"], expected)
        self.assertEqual(sum(report["safety"]["harmful_full_lengths"]["histogram"].values()), 20)
        self.assertEqual(sum(report["benign"]["full_lengths"]["histogram"].values()), 20)


if __name__ == "__main__":
    unittest.main()
