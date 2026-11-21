from pathlib import Path
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from architecture import Model
from tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "best_model.pt"
VAL_DATA = ROOT / "data" / "corpus" / "val.bin"

PROMPTS = [
    "Once upon a time,",
    "The main purpose of education is",
    "A computer program works by",
    "The weather today is",
    "Artificial intelligence is a field of computer science that",
    "Explain the difference between machine learning and deep learning.",
    "Write a Python function that checks whether a number is prime.",
]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = Config(**checkpoint["config"])
    tokenizer = Tokenizer()

    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab={tokenizer.get_vocab_size()}, "
            f"model vocab={config.vocab_size}"
        )

    model = Model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step: {checkpoint.get('step', 'unknown')}")
    print(f"Tokens seen: {checkpoint.get('tokens_seen', 'unknown'):,}")
    print(f"Recorded validation loss: "
          f"{checkpoint.get('val_loss', 'unknown')}")
    print(f"Parameters: {params:,}")
    print(f"Vocabulary: {config.vocab_size}")
    print(f"Device: {device}")

    return model, config, tokenizer


@torch.inference_mode()
def evaluate_loss(model, config, device, eval_tokens=1_000_000,
                  batch_size=4):
    if not VAL_DATA.exists():
        raise FileNotFoundError(f"Validation data not found: {VAL_DATA}")

    tokens = np.memmap(VAL_DATA, dtype=np.uint16, mode="r")
    n_tokens = min(len(tokens), eval_tokens)

    # Each sequence contains seq_len inputs and one extra target token.
    seq_len = min(1024, config.max_seq_len)
    n_sequences = (n_tokens - 1) // seq_len

    if n_sequences < 1:
        raise ValueError("Not enough validation tokens.")

    total_loss = 0.0
    total_targets = 0

    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    for first in range(0, n_sequences, batch_size):
        last = min(first + batch_size, n_sequences)
        starts = range(first, last)

        chunks = [
            np.asarray(
                tokens[i * seq_len:i * seq_len + seq_len + 1],
                dtype=np.int64,
            )
            for i in starts
        ]

        batch = torch.from_numpy(np.stack(chunks)).to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, config.vocab_size).float(),
                y.reshape(-1),
                reduction="sum",
            )

        total_loss += loss_sum.item()
        total_targets += y.numel()

        if (first // batch_size) % 100 == 0:
            print(
                f"\rValidation batches: {last}/{n_sequences}",
                end="",
                flush=True,
            )

    print()

    mean_loss = total_loss / total_targets
    perplexity = math.exp(mean_loss)

    print("\n--- Validation results ---")
    print(f"Tokens evaluated: {total_targets:,}")
    print(f"Cross-entropy loss: {mean_loss:.4f}")
    print(f"Perplexity: {perplexity:.3f}")

    return mean_loss


def sample_next_token(logits, tokenizer, temperature=0.8, top_k=50):
    logits = logits.float().clone()

    # Prevent generation of control tokens, except EOS.
    special_ids = tokenizer.get_special_token_ids()
    eos_id = tokenizer.eos_id

    for name, token_id in special_ids.items():
        if token_id < 0 or token_id >= logits.numel():
            continue
        if name == "eos" or token_id == eos_id:
            continue
        logits[token_id] = float("-inf")

    logits /= temperature

    k = min(top_k, logits.numel())
    values, _ = torch.topk(logits, k)
    threshold = values[-1]
    logits[logits < threshold] = float("-inf")

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).item()


@torch.inference_mode()
def generate(model, config, tokenizer, prompt, device,
             max_new_tokens=100):
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        return ""

    generated = list(input_ids)
    eos_id = tokenizer.eos_id

    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    for _ in range(max_new_tokens):
        context = generated[-config.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(x)[0, -1, :]

        next_id = sample_next_token(logits, tokenizer)
        generated.append(next_id)

        if eos_id >= 0 and next_id == eos_id:
            break

    return tokenizer.decode(generated[len(input_ids):])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--eval-tokens", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, tokenizer = load_model(args.checkpoint, device)

    evaluate_loss(
        model,
        config,
        device,
        eval_tokens=args.eval_tokens,
        batch_size=args.batch_size,
    )

    print("\n--- Generation evaluation ---")

    for i, prompt in enumerate(PROMPTS, 1):
        # Reset RNG so each run is reproducible.
        torch.manual_seed(1000 + i)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(1000 + i)

        output = generate(
            model,
            config,
            tokenizer,
            prompt,
            device,
            max_new_tokens=args.max_new_tokens,
        )

        print(f"\n[{i}/{len(PROMPTS)}]")
        print(f"PROMPT: {prompt}")
        print(f"GENERATED: {output}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()