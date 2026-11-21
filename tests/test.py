import torch

from architecture import Model
from config import Config
from tokenizer import Tokenizer


def main():
    # 1. Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Load tokenizer
    tokenizer = Tokenizer()
    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size()}")

    # 3. Match model vocabulary to tokenizer
    config = Config()
    actual_vocab_size = tokenizer.get_vocab_size()

    if config.vocab_size != actual_vocab_size:
        raise ValueError(
            f"Config vocab_size={config.vocab_size}, but "
            f"tokenizer vocab_size={actual_vocab_size}. "
            "Update config.py to match the tokenizer."
        )

    # 4. Initialize an untrained model
    model = Model(config).to(device)
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {parameter_count:,}")
    print("Model initialized successfully.")

    # 5. Test the forward pass
    prompt = "What is the weather in Chennai?"
    input_ids = tokenizer.encode(prompt)

    if not input_ids:
        raise ValueError("Prompt produced no tokens.")

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits = model(input_tensor)

    expected_shape = (
        1,
        len(input_ids),
        actual_vocab_size,
    )

    assert tuple(logits.shape) == expected_shape, (
        f"Expected {expected_shape}, got {tuple(logits.shape)}"
    )

    print(f"\nPrompt: {prompt}")
    print(f"Input token IDs: {input_ids}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print("Forward pass successful.")

    # 6. Generate random text autoregressively
    max_new_tokens = 40
    temperature = 1.0

    generated_ids = list(input_ids)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Respect the configured context length.
            context_ids = generated_ids[-config.max_seq_len :]

            context = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=device,
            )

            next_token_logits = model(context)[:, -1, :]
            next_token_logits = next_token_logits / temperature

            # Random sampling from the model's output distribution.
            probabilities = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            ).item()

            generated_ids.append(next_token)

            if next_token == tokenizer.eos_id and tokenizer.eos_id >= 0:
                break

    generated_text = tokenizer.decode(generated_ids)

    print("\n" + "=" * 50)
    print("GENERATED TEXT (UNTRAINED MODEL)")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)

    print("\nAll model loading and inference tests passed.")


if __name__ == "__main__":
    main()
