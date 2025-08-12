from huggingface_hub import snapshot_download
import urllib.request
import os

def download_model_from_hf(repo_id, out_dir):
    print(f"✅ Download {repo_id} to {out_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=out_dir,
        local_dir_use_symlinks=False,
        # endpoint="https://hf-mirror.com/"
    )

def download_file_from_google(url, filename):
    """Download a file from a given URL and save it as filename."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Saved to {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to download checkpoint from {url}") from e
    
# 模型仓库ID
repo_id_OnePoseViaGen = "ZhengGeng/OnePoseViaGen"

# output_dir
hf_model_dict = {
    repo_id_OnePoseViaGen : "checkpoints/OnePoseViaGen"
}

for key,value in hf_model_dict.items():
    download_model_from_hf(key, value)