import os
import yaml
import wget
import subprocess
import dotenv

dotenv.load_dotenv()
HF_USER = os.getenv("HF_USER")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_USER or not HF_TOKEN:
    raise ValueError("Hugging Face credentials are not set in environment variables.")


def download_direct(url, path_save):
    wget.download(url, path_save)
    print(f"\nDownloaded {url} to {path_save}")


def download_git(url, path_save):
    try:
        subprocess.run(["git", "clone", url, path_save])
    except Exception as e:
        print(f"Error downloading {url} to {path_save}: {e}")


def download_huggingface(url, path_save):
    try:
        data_url = url.split("//")[-1]
        repo_url = f"https://{HF_USER}:{HF_TOKEN}@{data_url}"
        subprocess.run(
            ["git", "clone", repo_url, path_save],
            check=True,
            text=True,
            capture_output=True
        )
        # Pull LFS files after cloning
        subprocess.run(
            ["git", "lfs", "pull"],
            cwd=path_save,
            check=True,
            text=True,
            capture_output=True
        )
    except Exception as e:
        print(f"Error downloading {url} to {path_save}: {e}")


if __name__ == "__main__":
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    available_datasets = list(config.keys())
    print("Available datasets:")
    for idx, dataset in enumerate(available_datasets, 1):
        print(f"{idx}. {dataset}")
    
    selected = input("Enter the datasets to download (comma-separated or 'all'): ").strip()
    
    if selected.lower() == "all":
        datasets_to_download = available_datasets
    else:
        selected_indexes = selected.split(',')
        datasets_to_download = [available_datasets[int(i)-1] for i in selected_indexes if i.isdigit() and 1 <= int(i) <= len(available_datasets)]
    
    for dataset in datasets_to_download:
        print(f"Downloading {dataset}...")
        path_save = config[dataset]['path_save']
        if not os.path.exists(path_save):
            if config[dataset].get('pre_cmd', None):
                subprocess.run(config[dataset]['pre_cmd'], shell=True)

            if config[dataset]['download_type'] == "direct":
                urls = config[dataset].get("urls", [config[dataset].get("url")])
                for url in urls:
                    if url:
                        download_direct(url, path_save)
            elif config[dataset]['download_type'] == "git":
                download_git(config[dataset]['url'], path_save)
            elif config[dataset]['download_type'] == "huggingface":
                download_huggingface(config[dataset]['url'], path_save)
            else:
                raise ValueError(f"Invalid download type: {config[dataset]['download_type']}")
            
            if config[dataset].get('cmd', None):
                for cmd in config[dataset]['cmd']:
                    subprocess.run(cmd, shell=True)
            
            # Remove .git folder if exists
            git_folder = os.path.join(path_save, '.git')
            if os.path.exists(git_folder):
                subprocess.run(['rm', '-rf', git_folder])
                print(f"Removed .git folder from {path_save}")
        else:
            print(f"Dataset {dataset} already exists in {path_save}. Skipping...")
    
    print("Downloading completed")