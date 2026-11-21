#!/usr/bin/env python3
"""Generate, validate, and dispatch tc-lm tool calls.

This file deliberately does not execute arbitrary code from tools.json.
You must register a Python callable for each tool you want to execute.
It prints the tool results; the current tc-lm SFT checkpoint is trained to
emit tool calls, not to write final natural-language answers from tool results.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from architecture import Model
from config import Config
from tokenizer import Tokenizer

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None


# Register real implementations here. Signatures receive keyword arguments.
# Replace these examples with your actual API/service integrations.
def get_weather(*, city: str) -> dict:
    raise NotImplementedError(
        "Connect get_weather to a real weather API before using this tool."
    )


def web_search(*, query: str) -> dict:
    raise NotImplementedError(
        "Connect web_search to a real search API before using this tool."
    )


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_weather": get_weather,
    "web_search": web_search,
}


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def extract_json(text):
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


def normalize_schema_type(schema):
    """Convert common simplified type names into standard JSON Schema types."""
    if isinstance(schema, list):
        return [normalize_schema_type(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    out = {}
    type_aliases = {
        "str": "string", "integer": "integer", "int": "integer",
        "float": "number", "double": "number", "bool": "boolean",
        "dict": "object", "list": "array",
    }
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = type_aliases.get(value, value)
        else:
            out[key] = normalize_schema_type(value)
    return out


def parameters_to_schema(parameters):
    """Accept standard JSON Schema or a simplified {arg: {type: ...}} map."""
    known_schema_keys = {
        "$schema", "$id", "$ref", "type", "properties", "required",
        "additionalProperties", "description", "title", "items", "enum",
        "oneOf", "anyOf", "allOf", "minimum", "maximum", "minLength",
        "maxLength", "pattern", "format",
    }
    if "properties" in parameters or set(parameters).issubset(known_schema_keys):
        schema = dict(parameters)
        schema.setdefault("type", "object")
        return normalize_schema_type(schema)

    # Simplified form used by some hand-written tool definitions:
    # {"query": {"type": "str", "description": "..."}}
    properties = {}
    required = []
    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            properties[name] = {"type": "string"}
            continue
        spec_copy = dict(spec)
        is_required = bool(spec_copy.pop("required", False))
        properties[name] = normalize_schema_type(spec_copy)
        if is_required:
            required.append(name)
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def validate_arguments(tool_def, arguments):
    """Validate arguments against standard or simplified parameter schemas."""
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    parameters = tool_def.get("parameters")
    if not isinstance(parameters, dict):
        # Some tool packs omit a schema; object type is still enforced above.
        return

    if Draft202012Validator is None:
        raise RuntimeError(
            "Install schema validation support first: pip install jsonschema"
        )

    schema = parameters_to_schema(parameters)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(arguments),
        key=lambda e: list(e.path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, e.path)) or '<root>'}: {e.message}"
            for e in errors[:5]
        )
        raise ValueError(f"Arguments failed schema validation: {details}")


def validate_calls(calls, tools):
    if not isinstance(calls, list) or not calls:
        raise ValueError("Model output must be a non-empty JSON array.")

    tool_defs = {
        t["name"]: t for t in tools
        if isinstance(t, dict) and isinstance(t.get("name"), str)
    }
    for i, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"Call {i} is not a JSON object.")
        name = call.get("name")
        if name not in tool_defs:
            raise ValueError(f"Call {i} uses an unlisted tool: {name!r}.")
        if not isinstance(call.get("arguments"), dict):
            raise ValueError(f"Call {i} arguments must be an object.")
        validate_arguments(tool_defs[name], call["arguments"])
    return tool_defs


def generate_calls(model, tokenizer, config, prompt, tools, device, max_new_tokens):
    prefix = (
        "<tools>\n"
        + compact_json(tools)
        + "\n"
        + f"User: {prompt.strip()}\n"
        + "Assistant: "
    )
    prompt_ids = tokenizer.encode(prefix, add_bos=True)
    if len(prompt_ids) >= config.max_seq_len:
        raise ValueError(
            f"Prompt is {len(prompt_ids)} tokens; max context is {config.max_seq_len}."
        )

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = ids[:, -config.max_seq_len:]
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(context)[:, -1, :]
            else:
                logits = model(context)[:, -1, :]
            next_id = logits.float().argmax(dim=-1, keepdim=True)
            token_id = int(next_id.item())
            if token_id == tokenizer.eos_id:
                break
            generated.append(token_id)
            ids = torch.cat((ids, next_id), dim=1)

    raw = tokenizer.decode(generated).strip()
    candidate = raw
    if "<tool_call>" in candidate:
        candidate = candidate.split("<tool_call>", 1)[1]
    if "</tool_call>" in candidate:
        candidate = candidate.split("</tool_call>", 1)[0]
    parsed = extract_json(candidate.strip())
    if parsed is None:
        raise ValueError(f"Model did not emit complete JSON. Raw output:\n{raw}")
    return raw, parsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--tools-file", type=Path)
    group.add_argument("--tools-json")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/tool_sft/best_model.pt"))
    p.add_argument("--tokenizer", type=Path,
                   default=Path("tokenizer/needle/tokenizer.model"))
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--execute", action="store_true",
                   help="Actually invoke registered Python functions.")
    args = p.parse_args()

    raw_tools = (
        args.tools_file.read_text(encoding="utf-8")
        if args.tools_file else args.tools_json
    )
    tools = json.loads(raw_tools)
    if not isinstance(tools, list) or not tools:
        raise ValueError("Tools must be a non-empty JSON array.")

    for path in (args.checkpoint, args.tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config(**checkpoint["config"]) if checkpoint.get("config") else Config()
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary and checkpoint config do not match.")

    model = Model(config)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("Checkpoint contains no model state dictionary.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    raw_output, calls = generate_calls(
        model, tokenizer, config, args.prompt, tools, device,
        args.max_new_tokens
    )
    print("\nRaw model output:\n" + raw_output)
    print("\nParsed calls:\n" + json.dumps(calls, ensure_ascii=False, indent=2))

    tool_defs = validate_calls(calls, tools)
    print("\nValidation: JSON structure, tool names, and available parameter schemas passed.")

    if not args.execute:
        print("\nDry run only. No tool was executed. Add --execute to dispatch registered tools.")
        return

    results = []
    for i, call in enumerate(calls):
        name = call["name"]
        function = TOOL_REGISTRY.get(name)
        if function is None:
            raise RuntimeError(
                f"No Python implementation registered for tool {name!r}. "
                "Add it to TOOL_REGISTRY before execution."
            )
        try:
            result = function(**call["arguments"])
            # Ensure result can be serialized and safely passed/logged as JSON.
            json.dumps(result, ensure_ascii=False)
            results.append({"name": name, "result": result, "error": None})
        except Exception as exc:
            results.append({
                "name": name,
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
            })

    print("\nTool results:\n" + json.dumps(results, ensure_ascii=False, indent=2))
    print(
        "\nExecution complete. This checkpoint was fine-tuned for tool-call "
        "generation; it has not been trained here to synthesize these results "
        "into a final user-facing answer."
    )


if __name__ == "__main__":
    main()
