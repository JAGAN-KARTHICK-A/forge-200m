
from pathlib import Path
import math

import torch
from config import Config
from architecture import Model
from tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"

PROMPTS = [
    "Once upon a time,",
    "The main purpose of education is",
    "A computer program works by",
    "The weather today is",
    "Artificial intelligence is a field of computer science that",
]

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.90


def load_checkpoint(path, device, tokenizer):
    print(f"\nLoading checkpoint: {path}")

    # Only load checkpoints you trust; these contain Python RNG state.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    config = Config(**checkpoint["config"])

    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab={tokenizer.get_vocab_size()}, "
            f"checkpoint vocab={config.vocab_size}"
        )

    model = Model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    step = checkpoint.get("step", "unknown")
    tokens_seen = checkpoint.get("tokens_seen", "unknown")
    val_loss = checkpoint.get(
        "val_loss",
        checkpoint.get("best_val_loss", None),
    )

    print(f"Step: {step}")
    print(f"Tokens seen: {tokens_seen}")

    if val_loss is not None:
        print(f"Recorded validation loss: {val_loss:.4f}")

    return model, config


def filter_logits(logits, generated_ids, tokenizer):
    # Avoid repeatedly emitting already-generated tokens.
    # This is deliberately mild so English repetition remains measurable.
    if generated_ids:
        recent = set(generated_ids[-64:])
        for token_id in recent:
            logits[token_id] -= 0.5

    # Do not generate control tokens in this plain-English evaluation.
    # EOS remains allowed so generation can stop naturally.
    special_ids = tokenizer.get_special_token_ids()
    eos_id = tokenizer.eos_id

    for name, token_id in special_ids.items():
        if token_id < 0 or token_id >= logits.numel():
            continue
        if name == "eos" or token_id == eos_id:
            continue
        logits[token_id] = float("-inf")

    return logits


def sample_next_token(logits, generated_ids, tokenizer):
    logits = logits.float() / TEMPERATURE
    logits = filter_logits(logits, generated_ids, tokenizer)

    # Top-k filtering
    if TOP_K > 0 and TOP_K < logits.numel():
        threshold = torch.topk(logits, TOP_K).values[-1]
        logits[logits < threshold] = float("-inf")

    # Top-p (nucleus) filtering
    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True
    )
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove = cumulative_probs > TOP_P
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    remove_indices = sorted_indices[remove]
    logits[remove_indices] = float("-inf")

    probs = torch.softmax(logits, dim=-1)

    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        raise RuntimeError("Invalid sampling probabilities.")

    return torch.multinomial(probs, num_samples=1).item()


@torch.inference_mode()
def generate(model, config, tokenizer, prompt, device):
    input_ids = tokenizer.encode(prompt)

    if not input_ids:
        raise ValueError(f"Prompt encoded to no tokens: {prompt!r}")

    generated_ids = list(input_ids)
    eos_id = tokenizer.eos_id

    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    for _ in range(MAX_NEW_TOKENS):
        context_ids = generated_ids[-config.max_seq_len:]
        x = torch.tensor(
            [context_ids],
            dtype=torch.long,
            device=device,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(x)[0, -1, :]

        next_id = sample_next_token(
            logits,
            generated_ids,
            tokenizer,
        )
        generated_ids.append(next_id)

        if eos_id >= 0 and next_id == eos_id:
            break

    completion_ids = generated_ids[len(input_ids):]
    return tokenizer.decode(completion_ids)


def main():
    torch.manual_seed(1234)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = Tokenizer()
    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size()}")

    checkpoints = [
        ("BEST CHECKPOINT", CHECKPOINT_DIR / "best_model.pt"),
        ("LATEST CHECKPOINT", CHECKPOINT_DIR / "latest.pt"),
    ]

    for label, path in checkpoints:
        if not path.is_file():
            print(f"\nSkipping {label}: {path} does not exist.")
            continue

        print("\n" + "=" * 80)
        print(label)
        print("=" * 80)

        model, config = load_checkpoint(path, device, tokenizer)

        for i, prompt in enumerate(PROMPTS, start=1):
            print(f"\n--- Sample {i} ---")
            print(f"PROMPT: {prompt}")

            completion = generate(
                model,
                config,
                tokenizer,
                prompt,
                device,
            )

            print(f"GENERATED: {completion}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
