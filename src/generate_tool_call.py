#!/usr/bin/env python3
"""Generate and validate tool calls with tc-lm."""

import argparse
import json
from pathlib import Path

import torch

from architecture import Model
from config import Config
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--tools-json")
    group.add_argument("--tools-file", type=Path)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/tool_sft_smoke/best_model.pt"))
    p.add_argument("--tokenizer", type=Path,
                   default=Path("tokenizer/needle/tokenizer.model"))
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_tools(args):
    raw = (args.tools_file.read_text(encoding="utf-8")
           if args.tools_file else args.tools_json)
    tools = json.loads(raw)
    if not isinstance(tools, list) or not tools:
        raise ValueError("Tool definitions must be a non-empty JSON list.")
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise ValueError(f"Tool at index {i} must have a string name.")
    return tools


def choose_next_token(logits, temperature, top_k, top_p):
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs >= top_p
        sorted_logits[remove] = float("-inf")
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits)
        logits = filtered

    return torch.multinomial(torch.softmax(logits, dim=-1), 1)


def extract_json(text):
    """Extract the first complete JSON array/object from generated text."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def validate_calls(value, tools):
    if not isinstance(value, list):
        return False, "Expected a JSON array of tool calls."
    available = {t["name"] for t in tools}
    if not value:
        return False, "Tool-call array is empty."
    for i, call in enumerate(value):
        if not isinstance(call, dict):
            return False, f"Call {i} is not an object."
        if call.get("name") not in available:
            return False, f"Call {i} uses an unknown tool: {call.get('name')!r}."
        if not isinstance(call.get("arguments"), dict):
            return False, f"Call {i} arguments must be a JSON object."
    return True, "JSON structure and tool names are valid."


@torch.inference_mode()
def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.tokenizer.is_file():
        raise FileNotFoundError(args.tokenizer)

    tools = load_tools(args)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(args.tokenizer)

    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)
    config = Config(**checkpoint["config"]) if checkpoint.get("config") else Config()
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary and model config do not match.")

    prefix = (
        "<tools>\n"
        + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        + f"User: {args.prompt.strip()}\n"
        + "Assistant: "
    )
    ids_list = tokenizer.encode(prefix, add_bos=True)
    if len(ids_list) >= config.max_seq_len:
        raise ValueError(
            f"Prompt uses {len(ids_list)} tokens; context limit is "
            f"{config.max_seq_len}. Shorten tools or prompt."
        )

    model = Model(config)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("Checkpoint has no model state dictionary.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    dtype = (torch.bfloat16 if device.type == "cuda"
             and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type == "cuda" else None)

    ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    generated_ids = []
    eos_id = tokenizer.eos_id

    for _ in range(args.max_new_tokens):
        context = ids[:, -config.max_seq_len:]
        if device.type == "cuda" and dtype is not None:
            with torch.autocast("cuda", dtype=dtype):
                logits = model(context)[:, -1, :]
        else:
            logits = model(context)[:, -1, :]

        next_id = choose_next_token(
            logits, args.temperature, args.top_k, args.top_p
        )
        token_id = int(next_id.item())
        if eos_id >= 0 and token_id == eos_id:
            break
        generated_ids.append(token_id)
        ids = torch.cat((ids, next_id), dim=1)

    output = tokenizer.decode(generated_ids).strip()
    print("Checkpoint:", args.checkpoint)
    print("Prompt:", args.prompt)
    print("\nRaw generated output:\n")
    print(output)

    # The model's generation is not changed or silently repaired.
    candidate = output
    if "<tool_call>" in candidate:
        candidate = candidate.split("<tool_call>", 1)[1]
    if "</tool_call>" in candidate:
        candidate = candidate.split("</tool_call>", 1)[0]
    candidate = candidate.strip()

    parsed = extract_json(candidate)
    print("\nValidation:")
    if parsed is None:
        print("- Invalid/incomplete JSON: no complete JSON object or array found.")
        print("- Do not execute this output.")
        return

    print("- Parsed JSON:")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    ok, message = validate_calls(parsed, tools)
    print(f"- {message}")
    if not ok:
        print("- Do not execute this output.")


if __name__ == "__main__":
    main()
