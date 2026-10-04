"""Validate private cold-email examples before they reach the trainer."""

import json
from pathlib import Path


def load_examples(paths):
    examples = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON array")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{path}[{index}]: expected an object")
            for key in ("prompt", "response"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}[{index}].{key}: expected nonempty text")
            examples.append({"prompt": row["prompt"].strip(), "response": row["response"].strip()})
    if len(examples) < 2:
        raise ValueError("At least two examples are needed for a train/evaluation split")
    return examples


def to_conversations(examples):
    return [
        {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "completion": [{"role": "assistant", "content": row["response"]}],
        }
        for row in examples
    ]
