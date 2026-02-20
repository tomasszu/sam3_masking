from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/sam3",
    local_dir="./sam3_weights",
    local_dir_use_symlinks=False
)