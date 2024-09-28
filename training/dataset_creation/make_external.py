"""Script to create the dataset."""

import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import requests
from loguru import logger
from tqdm import tqdm

from enfify import EXTERNAL_DATA_DIR

np.random.seed(42)


def download_whu_zip():
    raise NotImplementedError(
        "This function does not work yet because it doesn't distinguish between zip dirs and zip files properly."
    )
    repo_url = "https://github.com/ghua-ac/ENF-WHU-Dataset/archive/78ed7f3784949f769f291fc1cb94acd10da6322f.zip"
    dataset_path = "ENF-WHU-Dataset-78ed7f3784949f769f291fc1cb94acd10da6322f/ENF-WHU-Dataset"
    extract_path = EXTERNAL_DATA_DIR / "temp_zip"
    dataset_basename = os.path.basename(dataset_path)  # Get the basename

    # Stream the request to handle large files
    response = requests.get(repo_url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kilobyte

        # Download the file with a progress bar
        logger.debug("Downloading the repository")
        with io.BytesIO() as temp_file:
            progress_bar = tqdm(total=total_size_in_bytes, unit="B", unit_scale=True)

            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                temp_file.write(data)

            progress_bar.close()

            # If the progress bar didn't finish, show an error
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                logger.error("ERROR: Something went wrong during the download")
            else:
                logger.debug("Download completed")

            # Extract the specific folder from the downloaded ZIP
            temp_file.seek(0)  # Move back to the start of the file
            with zipfile.ZipFile(temp_file) as zip_file:
                os.makedirs(extract_path, exist_ok=True)

                # Extract only the specific folder with basename
                for member in zip_file.namelist():
                    if member.startswith(dataset_path):
                        # Compute the relative path within the dataset
                        relative_path = os.path.relpath(member, start=dataset_path)
                        extract_to = os.path.join(extract_path, dataset_basename, relative_path)
                        # Create necessary directories
                        if member.endswith("/"):  # It's a directory
                            os.makedirs(extract_to, exist_ok=True)
                        else:  # It's a file
                            os.makedirs(os.path.dirname(extract_to), exist_ok=True)
                            # Extract file
                            with zip_file.open(member) as source, open(extract_to, "wb") as target:
                                shutil.copyfileobj(source, target)

                logger.info(f"Folder {dataset_basename} extracted successfully to {extract_path}")
    else:
        logger.info(f"Failed to download the repository: {response.status_code}")


def download_whu_git():
    repo_url = "https://github.com/ghua-ac/ENF-WHU-Dataset.git"
    commit_hash = "78ed7f3784949f769f291fc1cb94acd10da6322f"
    specific_folders = ["ENF-WHU-Dataset/H1", "ENF-WHU-Dataset/H1_ref"]
    specific_folders_common_base = "ENF-WHU-Dataset"
    target_dir = EXTERNAL_DATA_DIR

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone the repository with a sparse-checkout filter
        logger.debug("Cloning repository")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", repo_url, temp_dir], check=True
        )

        # Change to the temporary directory
        os.chdir(temp_dir)

        # Set up sparse-checkout
        subprocess.run(["git", "sparse-checkout", "init"], check=True)

        # Define the paths to download
        with open(".git/info/sparse-checkout", "w") as sparse_file:
            for specific_folder in specific_folders:
                sparse_file.write(f"{specific_folder}/\n")

        logger.debug("Fetching content")
        subprocess.run(["git", "checkout", commit_hash], check=True)
        logger.debug("Content fetched")

        # Path to the specific folder
        specific_folder_path = Path(temp_dir) / specific_folders_common_base

        # Move the specific folder to the target directory
        if specific_folder_path.exists():
            shutil.move(str(specific_folder_path), str(target_dir))
            logger.info(f"Successfully moved {specific_folder_path} to {target_dir}")
        else:
            logger.info(f"Folder {specific_folder_path} not found.")

        # The temporary directory will be automatically deleted when the context manager ends


if __name__ == "__main__":
    download_whu_git()
