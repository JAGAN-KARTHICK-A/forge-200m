from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Cactus-Compute/needle",
    local_dir="tokenizer/needle",
    allow_patterns=[
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
)
