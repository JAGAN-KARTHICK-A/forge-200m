
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from config import Config
from tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "corpus"

REPO = "HuggingFaceTB/smollm-corpus"
CONFIG_NAME = "fineweb-edu-dedup"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-tokens", type=int, default=5_000_000)
    p.add_argument("--val-tokens", type=int, default=100_000)
    p.add_argument("--val-percent", type=int, default=2)
    p.add_argument("--max-docs", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=OUT)
    return p.parse_args()


def validation_document(text, percent):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100 < percent


def main():
    args = parse_args()

    if args.train_tokens <= 0 or args.val_tokens <= 0:
        raise ValueError("Token targets must be positive.")
    if not 1 <= args.val_percent <= 20:
        raise ValueError("--val-percent must be between 1 and 20.")

    config = Config()
    tokenizer = Tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    if vocab_size != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary={vocab_size}, "
            f"Config.vocab_size={config.vocab_size}"
        )

    if vocab_size > 65536:
        raise ValueError("Vocabulary cannot be represented by uint16.")

    eos_id = tokenizer.eos_id
    if eos_id < 0:
        eos_id = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.bin"
    val_path = args.output_dir / "val.bin"

    print(f"Streaming {REPO} / {CONFIG_NAME}")
    print(f"Tokenizer vocabulary: {vocab_size}")
    print(f"Targets: train={args.train_tokens:,}, val={args.val_tokens:,}")

    ds = load_dataset(
        REPO,
        CONFIG_NAME,
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=10_000)

    train_n = 0
    val_n = 0
    docs = 0

    # New files for each fresh preparation run.
    with train_path.open("wb") as tf, val_path.open("wb") as vf:
        progress = tqdm(
            total=args.train_tokens + args.val_tokens,
            desc="Preparing token corpus",
            unit="tokens",
        )

        for row in ds:
            docs += 1
            text = row.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            is_val = validation_document(text, args.val_percent)

            if is_val and val_n >= args.val_tokens:
                continue
            if not is_val and train_n >= args.train_tokens:
                continue

            # Avoid huge allocations from unusually large web pages.
            ids = tokenizer.encode(text[:200_000])
            if not ids:
                continue

            if eos_id is not None:
                ids.append(eos_id)

            if is_val:
                remaining = args.val_tokens - val_n
                ids = ids[:remaining]
                np.asarray(ids, dtype=np.uint16).tofile(vf)
                val_n += len(ids)
            else:
                remaining = args.train_tokens - train_n
                ids = ids[:remaining]
                np.asarray(ids, dtype=np.uint16).tofile(tf)
                train_n += len(ids)

            progress.update(len(ids))
            if docs % 1000 == 0:
                progress.set_postfix(
                    train=f"{train_n:,}",
                    val=f"{val_n:,}",
                    docs=docs,
                )

            if train_n >= args.train_tokens and val_n >= args.val_tokens:
                break

            if docs >= args.max_docs:
                break

        progress.close()

    metadata = {
        "dataset": REPO,
        "configuration": CONFIG_NAME,
        "tokenizer": "Cactus-Compute/needle",
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "train_tokens": train_n,
        "val_tokens": val_n,
        "documents_seen": docs,
        "seed": args.seed,
    }

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if train_n < args.train_tokens or val_n < args.val_tokens:
        raise RuntimeError(
            f"Corpus incomplete: train={train_n:,}/{args.train_tokens:,}, "
            f"validation={val_n:,}/{args.val_tokens:,}. "
            "Rerun with a larger --max-docs."
        )

    print(f"Training file: {train_path} ({train_n:,} tokens)")
    print(f"Validation file: {val_path} ({val_n:,} tokens)")
    print("Corpus preparation complete.")


if __name__ == "__main__":
    main()
