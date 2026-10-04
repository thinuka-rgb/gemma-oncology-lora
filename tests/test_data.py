import json
import tempfile
import unittest
from pathlib import Path

from data import load_examples, to_conversations


class DataTests(unittest.TestCase):
    def test_multiple_files_and_conversational_format(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(2):
                path = Path(directory) / f"{i}.json"
                path.write_text(json.dumps([{"prompt": f"Question {i}", "response": f"Answer {i}"}]), encoding="utf-8")
                paths.append(path)
            rows = load_examples(paths)
            self.assertEqual(len(rows), 2)
            self.assertEqual(to_conversations(rows)[0], {
                "prompt": [{"role": "user", "content": "Question 0"}],
                "completion": [{"role": "assistant", "content": "Answer 0"}],
            })

    def test_rejects_missing_response(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps([{"prompt": "Only a prompt"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "response"):
                load_examples([path])


if __name__ == "__main__":
    unittest.main()
