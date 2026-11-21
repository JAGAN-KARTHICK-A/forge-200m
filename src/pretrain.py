
from pathlib import Path
from dataclasses import asdict
import argparse
import contextlib
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from architecture import Model
from tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "corpus"
CKPT_DIR = ROOT / "checkpoints"


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", type=Path, default=DATA_DIR / "train.bin")
    p.add_argument("--val-file", type=Path, default=DATA_DIR / "val.bin")
    p.add_argument("--checkpoint-dir", type=Path, default=CKPT_DIR)

    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=5_000_000)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", nargs="?", const="latest", default=None)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_tokens(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run download_ds_corpus.py first."
        )
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) < 2:
        raise ValueError(f"Not enough tokens in {path}")
    return data


def batch(data, batch_size, seq_len, device, rng):
    max_start = len(data) - seq_len - 1
    if max_start < 0:
        raise ValueError(f"Not enough tokens for sequence length {seq_len}")

    starts = rng.integers(0, max_start + 1, size=batch_size)
    rows = np.stack([
        np.asarray(data[s:s + seq_len + 1], dtype=np.int64)
        for s in starts
    ])

    x = torch.from_numpy(rows[:, :-1].copy()).to(device)
    y = torch.from_numpy(rows[:, 1:].copy()).to(device)
    return x, y


def lr_at(step, total_steps, args):
    if step <= args.warmup_steps:
        return args.lr * step / max(1, args.warmup_steps)

    progress = min(
        1.0,
        (step - args.warmup_steps)
        / max(1, total_steps - args.warmup_steps),
    )
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * (
        args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine
    )


def amp_dtype(device):
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def amp_context(device, dtype):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def loss_for(model, x, y, device, dtype):
    with amp_context(device, dtype):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
        )
    return loss


@torch.no_grad()
def evaluate(model, data, args, device, dtype):
    model.eval()
    rng = np.random.default_rng(args.seed + 999)
    losses = []

    for _ in range(args.eval_batches):
        x, y = batch(data, args.batch_size, args.seq_len, device, rng)
        losses.append(loss_for(model, x, y, device, dtype).item())

    model.train()
    return float(np.mean(losses))


def save_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def make_checkpoint(
    model, optimizer, scaler, step, tokens_seen, best_val,
    config, args, rng,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val,
        "config": asdict(config),
        "train_rng_state": rng.bit_generator.state,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    return payload


def main():
    args = arguments()
    set_seed(args.seed)

    config = Config()
    tokenizer = Tokenizer()

    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab={tokenizer.get_vocab_size()}, "
            f"config vocab={config.vocab_size}"
        )
    if args.seq_len > config.max_seq_len:
        raise ValueError("seq-len exceeds Config.max_seq_len")

    train_data = read_tokens(args.train_file)
    val_data = read_tokens(args.val_file)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this instance.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dtype = amp_dtype(device)
    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    model = Model(config).to(device)

    # Decay matrix weights; exclude biases and normalization parameters.
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    try:
        optimizer = torch.optim.AdamW(
            groups, lr=args.lr, betas=(0.9, 0.95),
            eps=1e-8, fused=True,
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        )

    train_model = model
    if args.compile:
        # Compile is optional; benchmark startup and steady-state speed.
        train_model = torch.compile(model)

    tokens_per_step = (
        args.batch_size * args.seq_len * args.grad_accum
    )
    total_steps = math.ceil(args.max_tokens / tokens_per_step)

    rng = np.random.default_rng(args.seed)
    step = 0
    tokens_seen = 0
    best_val = float("inf")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        path = (
            args.checkpoint_dir / "latest.pt"
            if args.resume == "latest"
            else Path(args.resume)
        )
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )

        if checkpoint["config"] != asdict(config):
            raise ValueError("Checkpoint config differs from current Config.")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])

        step = checkpoint["step"]
        tokens_seen = checkpoint["tokens_seen"]
        best_val = checkpoint["best_val_loss"]
        rng.bit_generator.state = checkpoint["train_rng_state"]

        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        print(f"Resumed from {path} at step {step}")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    print(f"Precision: {dtype}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Tokens per optimizer step: {tokens_per_step:,}")
    print(f"Target tokens: {args.max_tokens:,}; steps: {total_steps}")

    train_model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    started = time.time()

    for step_now in range(step + 1, total_steps + 1):
        lr = lr_at(step_now, total_steps, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        accumulated_loss = 0.0

        for _ in range(args.grad_accum):
            x, y = batch(
                train_data, args.batch_size, args.seq_len, device, rng
            )
            loss = loss_for(train_model, x, y, device, dtype)
            scaler.scale(loss / args.grad_accum).backward()
            accumulated_loss += loss.detach().item() / args.grad_accum

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += tokens_per_step
        running_loss += accumulated_loss

        if step_now % args.log_interval == 0 or step_now == 1:
            avg = running_loss / min(args.log_interval, step_now)
            elapsed = time.time() - started
            print(
                f"step={step_now}/{total_steps} "
                f"tokens={tokens_seen:,} loss={avg:.4f} "
                f"lr={lr:.2e} grad_norm={float(grad_norm):.3f} "
                f"elapsed={elapsed:.1f}s"
            )
            running_loss = 0.0

        if step_now % args.eval_interval == 0 or step_now == total_steps:
            val_loss = evaluate(model, val_data, args, device, dtype)
            print(f"[validation] step={step_now} loss={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                save_atomic({
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "step": step_now,
                    "tokens_seen": tokens_seen,
                    "val_loss": val_loss,
                }, args.checkpoint_dir / "best_model.pt")

        if step_now % args.save_interval == 0 or step_now == total_steps:
            payload = make_checkpoint(
                model, optimizer, scaler, step_now, tokens_seen,
                best_val, config, args, rng,
            )
            save_atomic(payload, args.checkpoint_dir / "latest.pt")
            print(f"Saved checkpoint at step {step_now}")

    print(f"Finished. Tokens seen: {tokens_seen:,}")
    print(f"Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
