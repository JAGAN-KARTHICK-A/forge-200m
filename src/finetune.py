#!/usr/bin/env python3
"""Supervised fine-tuning for tc-lm on xLAM function-calling records."""

import argparse
import contextlib
import json
import math
import random
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from architecture import Model
from config import Config
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True,
                   help="Downloaded xLAM JSON array or JSONL file.")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/best_model.pt"),
                   help="Pretrained tc-lm checkpoint.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("checkpoints/tool_sft"))
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true",
                   help="Resume from OUTPUT_DIR/latest.pt.")
    p.add_argument("--limit-records", type=int, default=0,
                   help="Optional small subset for a smoke test; 0 uses all records.")
    return p.parse_args()


def read_source(path):
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dataset file is empty: {path}")

    # Accept a JSON array/object or newline-delimited JSON.
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict) and isinstance(obj.get("train"), list):
            rows = obj["train"]
        elif isinstance(obj, dict):
            rows = [obj]
        else:
            raise ValueError("Unsupported JSON dataset structure.")
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]

    if not rows:
        raise ValueError("No records found in dataset.")
    return rows


def parse_json_field(value, field_name):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {field_name}") from exc
    return value


def normalize_record(row):
    query = row.get("query")
    tools = parse_json_field(row.get("tools"), "tools")
    answers = parse_json_field(row.get("answers"), "answers")

    if not isinstance(query, str) or not query.strip():
        return None
    if not isinstance(tools, list) or not tools:
        return None
    if isinstance(answers, dict):
        answers = [answers]
    if not isinstance(answers, list) or not answers:
        return None

    tool_names = {
        tool.get("name")
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }
    if not tool_names:
        return None

    for answer in answers:
        if not isinstance(answer, dict):
            return None
        if answer.get("name") not in tool_names:
            return None
        if not isinstance(answer.get("arguments", {}), dict):
            return None

    return {
        "query": query.strip(),
        "tools": tools,
        "answers": answers,
    }


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_example(record, tokenizer, seq_len):
    tools_text = compact_json(record["tools"])
    answers_text = compact_json(record["answers"])

    # Prompt tokens are masked from loss. The model must learn to emit
    # <tool_call> and the call JSON after the assistant prefix.
    prefix = (
        "<tools>\n"
        f"{tools_text}\n"
        f"User: {record['query']}\n"
        "Assistant: "
    )
    target = "<tool_call>\n" + answers_text

    # Only emit a closing marker if the local SentencePiece model really
    # contains it as a dedicated piece. EOS is appended by Tokenizer.
    if "</tool_call>" in tokenizer.special_tokens:
        target += "\n</tool_call>"

    prefix_ids = tokenizer.encode(prefix, add_bos=True)
    target_ids = tokenizer.encode(target, add_eos=True)
    if not target_ids:
        return None

    ids = prefix_ids + target_ids
    if len(ids) > seq_len:
        return None

    labels = [-100] * len(prefix_ids) + target_ids
    return (
        np.asarray(ids, dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
    )


def split_records(records, seed):
    # Exact duplicates are removed before splitting to reduce leakage.
    unique = []
    seen = set()
    duplicates = 0
    for record in records:
        key = compact_json({
            "query": record["query"],
            "tools": record["tools"],
            "answers": record["answers"],
        })
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(record)

    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, int(n * 0.05))
    n_val = max(1, int(n * 0.05))
    if n - n_test - n_val < 1:
        raise ValueError("Not enough unique records to create train/val/test splits.")

    test = unique[:n_test]
    val = unique[n_test:n_test + n_val]
    train = unique[n_test + n_val:]
    return train, val, test, duplicates


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def encode_records(records, tokenizer, seq_len):
    encoded = []
    too_long = 0
    invalid = 0
    for record in records:
        try:
            item = make_example(record, tokenizer, seq_len)
        except Exception as exc:
            invalid += 1
            if invalid <= 3:
                print(f"Skipping record due to tokenization error: {exc}")
            continue
        if item is None:
            too_long += 1
            continue
        encoded.append(item)
    return encoded, too_long, invalid


def collate(examples, pad_id, device):
    max_len = max(len(ids) - 1 for ids, _ in examples)
    x = torch.full((len(examples), max_len), pad_id,
                   dtype=torch.long)
    y = torch.full((len(examples), max_len), -100,
                   dtype=torch.long)

    for i, (ids, labels) in enumerate(examples):
        input_ids = ids[:-1]
        target_labels = labels[1:]
        n = len(input_ids)
        x[i, :n] = torch.from_numpy(input_ids.astype(np.int64, copy=False))
        y[i, :n] = torch.from_numpy(target_labels.astype(np.int64, copy=False))

    return x.to(device), y.to(device)


def get_amp(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device, dtype):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, examples, args, tokenizer, device, dtype):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = min(args.eval_batches,
                    math.ceil(len(examples) / args.batch_size))
    for batch_idx in range(n_batches):
        batch_examples = examples[
            batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size
        ]
        x, y = collate(batch_examples, tokenizer.pad_id
                       if tokenizer.pad_id >= 0 else 0, device)
        with autocast_context(device, dtype):
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        count = int((y != -100).sum().item())
        total_loss += float(loss_sum.item())
        total_tokens += count
    model.train()
    return total_loss / max(1, total_tokens)


def save_checkpoint(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def learning_rate(step, total_steps, base_lr, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = min(1.0, (step - warmup_steps) /
                   max(1, total_steps - warmup_steps))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset file not found: {args.data}")
    if not args.checkpoint.is_file() and not args.resume:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.checkpoint}")

    tokenizer = Tokenizer()
    config = Config()
    base_path = args.output_dir / "latest.pt" if args.resume else args.checkpoint
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)

    checkpoint_config = checkpoint.get("config")
    if checkpoint_config:
        config = Config(**checkpoint_config)
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab={tokenizer.get_vocab_size()}, "
            f"model config vocab={config.vocab_size}"
        )
    if args.seq_len > config.max_seq_len:
        raise ValueError(
            f"--seq-len {args.seq_len} exceeds model max_seq_len "
            f"{config.max_seq_len}"
        )

    print("Tokenizer special token IDs:", tokenizer.get_special_token_ids())
    for marker in ("<tools>", "<tool_call>"):
        if marker not in tokenizer.special_tokens:
            print(
                f"WARNING: {marker} is not a dedicated SentencePiece token; "
                "it will be tokenized as ordinary pieces."
            )

    model = Model(config)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("Checkpoint does not contain a 'model' state dict.")
    model.load_state_dict(state, strict=True)

    raw_rows = read_source(args.data)
    if args.limit_records > 0:
        raw_rows = raw_rows[:args.limit_records]
    records = [normalize_record(row) for row in raw_rows]
    records = [r for r in records if r is not None]
    if not records:
        raise ValueError("No valid xLAM records found.")

    train_records, val_records, test_records, duplicates = split_records(
        records, args.seed
    )
    split_dir = args.output_dir / "splits"
    write_jsonl(split_dir / "validation.jsonl", val_records)
    write_jsonl(split_dir / "test.jsonl", test_records)

    train_data, train_long, train_invalid = encode_records(
        train_records, tokenizer, args.seq_len
    )
    val_data, val_long, val_invalid = encode_records(
        val_records, tokenizer, args.seq_len
    )
    if not train_data or not val_data:
        raise ValueError(
            "No training or validation examples remain after length/format checks."
        )

    print(f"Valid unique records: {len(records):,}")
    print(f"Exact duplicates removed: {duplicates:,}")
    print(f"Train examples retained: {len(train_data):,}")
    print(f"Validation examples retained: {len(val_data):,}")
    print(f"Test examples saved: {len(test_records):,}")
    print(f"Too long and skipped: train={train_long:,}, val={val_long:,}")
    print(f"Tokenization errors: train={train_invalid:,}, val={val_invalid:,}")
    print(f"Sequence limit: {args.seq_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model.to(device)

    dtype = get_amp(device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and dtype == torch.float16)
    )

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            betas=(0.9, 0.95), weight_decay=args.weight_decay,
            fused=(device.type == "cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            betas=(0.9, 0.95), weight_decay=args.weight_decay,
        )

    step = 0
    best_val = float("inf")
    start_epoch = 0
    resume_batch = 0

    if args.resume:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        saved_epoch = int(checkpoint.get("epoch", 0))
        saved_batch = int(checkpoint.get("next_batch_idx", 0))
        n_batches = math.ceil(len(train_data) / args.batch_size)
        if saved_batch >= n_batches:
            start_epoch = saved_epoch + 1
            resume_batch = 0
        else:
            start_epoch = saved_epoch
            resume_batch = saved_batch
        print(f"Resuming from {base_path} at step {step}")

    batches_per_epoch = math.ceil(len(train_data) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    pad_id = tokenizer.pad_id if tokenizer.pad_id >= 0 else 0

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    print(f"Precision: {dtype}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Effective example batch: {args.batch_size * args.grad_accum}")
    print(f"Optimizer steps per epoch: {steps_per_epoch}")
    print(f"Output: {args.output_dir.resolve()}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(len(train_data))
        first_batch = resume_batch if epoch == start_epoch else 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx in range(first_batch, batches_per_epoch):
            indexes = order[
                batch_idx * args.batch_size:
                min((batch_idx + 1) * args.batch_size, len(order))
            ]
            examples = [train_data[int(i)] for i in indexes]
            x, y = collate(examples, pad_id, device)

            group_start = (batch_idx // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, batches_per_epoch - group_start)

            with autocast_context(device, dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                    ignore_index=-100,
                )

            scaler.scale(loss / group_size).backward()

            should_step = (
                (batch_idx + 1) % args.grad_accum == 0
                or batch_idx + 1 == batches_per_epoch
            )
            if not should_step:
                continue

            next_step = step + 1
            lr = learning_rate(next_step, total_steps, args.lr, args.warmup_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step = next_step

            if step == 1 or step % 10 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{total_steps} "
                    f"loss={loss.item():.4f} lr={lr:.2e}"
                )

            if step % args.eval_interval == 0 or batch_idx + 1 == batches_per_epoch:
                val_loss = evaluate(
                    model, val_data, args, tokenizer, device, dtype
                )
                print(f"validation step={step}: loss={val_loss:.4f}")

                payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "next_batch_idx": batch_idx + 1,
                    "best_val_loss": min(best_val, val_loss),
                    "config": vars(config),
                    "base_checkpoint": str(args.checkpoint),
                    "args": {k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()},
                }
                if val_loss < best_val:
                    best_val = val_loss
                    payload["best_val_loss"] = best_val
                    save_checkpoint(payload, args.output_dir / "best_model.pt")
                    print(f"Saved new best checkpoint (val loss {best_val:.4f})")

                if step % args.save_interval == 0 or batch_idx + 1 == batches_per_epoch:
                    payload["best_val_loss"] = best_val
                    save_checkpoint(payload, args.output_dir / "latest.pt")
                    print("Saved latest checkpoint")

            # Save latest independently of evaluation cadence, so a save
            # interval that falls between evaluation steps is still honored.
            if (step % args.save_interval == 0
                    and step % args.eval_interval != 0
                    and batch_idx + 1 != batches_per_epoch):
                payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "next_batch_idx": batch_idx + 1,
                    "best_val_loss": best_val,
                    "config": vars(config),
                    "base_checkpoint": str(args.checkpoint),
                    "args": {k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()},
                }
                save_checkpoint(payload, args.output_dir / "latest.pt")
                print("Saved latest checkpoint")

        resume_batch = 0

    print(f"Fine-tuning complete. Best validation loss: {best_val:.4f}")
    print(f"Best model: {args.output_dir / 'best_model.pt'}")
    print(f"Latest model: {args.output_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
