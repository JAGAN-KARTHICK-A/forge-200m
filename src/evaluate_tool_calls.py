#!/usr/bin/env python3
"""Held-out evaluation for the tc-lm tool-call checkpoint.

Run from the tc-lm project root. Uses the exact saved test split produced by
finetune.py, so it does not accidentally recreate a different split.
"""

import argparse
import json
from pathlib import Path

import torch

from architecture import Model
from config import Config
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/tool_sft/best_model.pt"))
    p.add_argument("--test-file", type=Path,
                   default=Path("checkpoints/tool_sft/splits/test.jsonl"))
    p.add_argument("--tokenizer", type=Path,
                   default=Path("tokenizer/needle/tokenizer.model"))
    p.add_argument("--limit", type=int, default=100,
                   help="Number of test records; use 0 for all.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--output", type=Path,
                   default=Path("checkpoints/tool_sft/evaluation.jsonl"))
    return p.parse_args()


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def extract_json(text):
    """Parse the first complete JSON value without repairing model output."""
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def validate_calls(calls, tools):
    if not isinstance(calls, list) or not calls:
        return False
    allowed = {
        tool.get("name") for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }
    for call in calls:
        if not isinstance(call, dict):
            return False
        if call.get("name") not in allowed:
            return False
        if not isinstance(call.get("arguments"), dict):
            return False
    return True


def build_prefix(record):
    return (
        "<tools>\n"
        + compact_json(record["tools"])
        + "\n"
        + f"User: {record['query']}\n"
        + "Assistant: "
    )


@torch.inference_mode()
def generate(model, tokenizer, config, prefix, device, max_new_tokens):
    prompt_ids = tokenizer.encode(prefix, add_bos=True)
    if len(prompt_ids) >= config.max_seq_len:
        return "", "prompt_too_long"

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    eos_id = tokenizer.eos_id
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for _ in range(max_new_tokens):
        context = ids[:, -config.max_seq_len:]
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(context)[:, -1, :]
        else:
            logits = model(context)[:, -1, :]

        next_id = logits.float().argmax(dim=-1, keepdim=True)
        token_id = int(next_id.item())
        if eos_id >= 0 and token_id == eos_id:
            break
        generated.append(token_id)
        ids = torch.cat((ids, next_id), dim=1)

    text = tokenizer.decode(generated).strip()
    candidate = text
    if "<tool_call>" in candidate:
        candidate = candidate.split("<tool_call>", 1)[1]
    if "</tool_call>" in candidate:
        candidate = candidate.split("</tool_call>", 1)[0]
    return candidate.strip(), text


def main():
    args = parse_args()
    for path in (args.checkpoint, args.test_file, args.tokenizer):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "The test split should be created by the original finetune.py run."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(args.tokenizer)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = Config(**checkpoint["config"]) if checkpoint.get("config") else Config()
    model = Model(config)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("Checkpoint has no model state dictionary.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    with args.test_file.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        records = records[:args.limit]
    if not records:
        raise ValueError(f"No records found in {args.test_file}")

    json_valid = 0
    structurally_valid = 0
    tool_sequence_exact = 0
    arguments_exact = 0
    calls_exact = 0
    total_expected_calls = 0
    total_predicted_calls = 0
    total_examples = len(records)
    results = []

    for idx, record in enumerate(records, 1):
        candidate, raw_output = generate(
            model, tokenizer, config, build_prefix(record), device,
            args.max_new_tokens
        )
        predicted = extract_json(candidate)
        expected = record["answers"]
        if isinstance(expected, dict):
            expected = [expected]

        is_json = predicted is not None and isinstance(predicted, list)
        if is_json:
            json_valid += 1

        is_valid = is_json and validate_calls(predicted, record["tools"])
        if is_valid:
            structurally_valid += 1

        expected_names = [c.get("name") for c in expected]
        predicted_names = (
            [c.get("name") for c in predicted]
            if isinstance(predicted, list) and all(isinstance(c, dict) for c in predicted)
            else []
        )
        if predicted_names == expected_names:
            tool_sequence_exact += 1

        # Compare arguments position-by-position only when the call names match.
        if len(predicted_names) == len(expected_names) and predicted_names == expected_names:
            if all(
                isinstance(predicted[i].get("arguments"), dict)
                and predicted[i].get("arguments") == expected[i].get("arguments", {})
                for i in range(len(expected))
            ):
                arguments_exact += 1

        if predicted == expected:
            calls_exact += 1

        total_expected_calls += len(expected)
        total_predicted_calls += len(predicted) if isinstance(predicted, list) else 0

        results.append({
            "index": idx,
            "query": record["query"],
            "expected": expected,
            "raw_output": raw_output,
            "parsed_prediction": predicted,
            "json_array_valid": bool(is_json),
            "structurally_valid": bool(is_valid),
            "tool_sequence_exact": predicted_names == expected_names,
            "arguments_exact_given_tool_sequence": (
                len(predicted_names) == len(expected_names)
                and predicted_names == expected_names
                and all(
                    isinstance(predicted[i].get("arguments"), dict)
                    and predicted[i].get("arguments") == expected[i].get("arguments", {})
                    for i in range(len(expected))
                )
            ),
            "exact_calls_match": predicted == expected,
        })

        if idx % 10 == 0 or idx == total_examples:
            print(f"Evaluated {idx}/{total_examples}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pct = lambda n, d: 100.0 * n / d if d else 0.0
    print("\n=== tc-lm held-out evaluation ===")
    print(f"Checkpoint:                    {args.checkpoint}")
    print(f"Test file:                     {args.test_file}")
    print(f"Examples evaluated:             {total_examples}")
    print(f"Valid JSON array:               {json_valid}/{total_examples} ({pct(json_valid, total_examples):.2f}%)")
    print(f"Valid structure/tool names:     {structurally_valid}/{total_examples} ({pct(structurally_valid, total_examples):.2f}%)")
    print(f"Exact tool-name sequence:       {tool_sequence_exact}/{total_examples} ({pct(tool_sequence_exact, total_examples):.2f}%)")
    print(f"Exact arguments, names match:   {arguments_exact}/{total_examples} ({pct(arguments_exact, total_examples):.2f}%)")
    print(f"Exact full call arrays:          {calls_exact}/{total_examples} ({pct(calls_exact, total_examples):.2f}%)")
    print(f"Expected calls:                 {total_expected_calls}")
    print(f"Predicted calls:                {total_predicted_calls}")
    print(f"Per-example details:             {args.output}")


if __name__ == "__main__":
    main()
