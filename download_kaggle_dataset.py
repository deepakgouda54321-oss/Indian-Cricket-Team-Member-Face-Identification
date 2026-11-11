"""
Kaggle and Google Dataset Search Downloader & Importer for Indian Cricketer Datasets.

Recommended Datasets:
1. 'shilpeeprasad/indian-cricketers-image-dataset' (~1,400 images of Indian cricketers)
2. 'aminulhossain01/indian-cricketers-dataset'
3. 'pranavsurte/cricket-players-faces'

Usage:
  - If you have Kaggle API token (~/.kaggle/kaggle.json):
      python download_kaggle_dataset.py --download shilpeeprasad/indian-cricketers-image-dataset
  - If you downloaded a zip manually:
      python download_kaggle_dataset.py --extract path/to/archive.zip
  - To reorganize and normalize existing folders in data/raw:
      python download_kaggle_dataset.py --clean-names
"""

import os
import sys
import shutil
import zipfile
import argparse
from typing import List


RECOMMENDED_DATASETS = [
    {
        "id": "shilpeeprasad/indian-cricketers-image-dataset",
        "title": "Indian Cricketers Image Dataset (Shilpee Prasad)",
        "url": "https://www.kaggle.com/datasets/shilpeeprasad/indian-cricketers-image-dataset",
        "notes": "Over 1,400 images partitioned by player name (approx 50 images per cricketer)."
    },
    {
        "id": "aminulhossain01/indian-cricketers-dataset",
        "title": "Indian Cricketers Dataset (Aminul Hossain)",
        "url": "https://www.kaggle.com/datasets/aminulhossain01/indian-cricketers-dataset",
        "notes": "Comprehensive photo gallery of prominent Indian men and women cricketers."
    },
    {
        "id": "pranavsurte/cricket-players-faces",
        "title": "Cricket Players Faces (Pranav Surte)",
        "url": "https://www.kaggle.com/datasets/pranavsurte/cricket-players-faces",
        "notes": "Crop-focused dataset of top cricket players prepared for CNN face classification."
    }
]


def normalize_player_name(folder_name: str) -> str:
    """Standardizes names like 'Virat Kohli' -> 'virat_kohli'."""
    clean = folder_name.strip().lower()
    for ch in ["-", " "]:
        clean = clean.replace(ch, "_")
    # Remove leading numbering if present (e.g. '01_virat_kohli' -> 'virat_kohli')
    parts = clean.split("_")
    if parts and parts[0].isdigit():
        clean = "_".join(parts[1:])
    return clean


def clean_raw_folder(raw_dir: str = "data/raw"):
    """
    Scans data/raw, normalizes folder names, handles nested archives,
    and removes non-image files.
    """
    if not os.path.exists(raw_dir):
        print(f"[!] Directory '{raw_dir}' does not exist.")
        return

    print(f"[*] Scanning and standardizing folders in '{raw_dir}'...")
    entries = os.listdir(raw_dir)

    for entry in entries:
        src_path = os.path.join(raw_dir, entry)
        if not os.path.isdir(src_path):
            continue

        normalized = normalize_player_name(entry)
        dest_path = os.path.join(raw_dir, normalized)

        if src_path != dest_path:
            if os.path.exists(dest_path):
                # Merge contents
                for f in os.listdir(src_path):
                    s_f = os.path.join(src_path, f)
                    d_f = os.path.join(dest_path, f)
                    if not os.path.exists(d_f):
                        shutil.move(s_f, d_f)
                shutil.rmtree(src_path)
            else:
                os.rename(src_path, dest_path)
            print(f"  [+] Renamed '{entry}' -> '{normalized}'")

    print("[+] Folder normalization complete.")


def extract_zip(zip_path: str, target_dir: str = "data/raw"):
    """Extracts a downloaded zip archive into target_dir."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    print(f"[*] Extracting '{zip_path}' into '{target_dir}'...")
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

    clean_raw_folder(target_dir)
    print(f"[+] Archive extracted and organized in '{target_dir}'!")


def download_with_kaggle_api(dataset_slug: str, target_dir: str = "data/raw"):
    """Attempts to download dataset using Kaggle CLI / API."""
    import subprocess
    print(f"[*] Attempting Kaggle CLI download for: {dataset_slug}")
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", target_dir, "--unzip"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
        clean_raw_folder(target_dir)
        print(f"[+] Successfully downloaded and unzipped {dataset_slug} into {target_dir}")
    except FileNotFoundError:
        print("\n[!] 'kaggle' CLI is not found in PATH.")
        print("    Install it via: pip install kaggle")
        print("    And place your API credentials in: ~/.kaggle/kaggle.json")
        print("\nAlternatively, download manually via browser from:")
        print(f"    https://www.kaggle.com/datasets/{dataset_slug}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Kaggle command failed:\n{e.stderr}")
        print("Verify your ~/.kaggle/kaggle.json API token and network connection.")


def print_guide():
    """Prints step-by-step dataset acquisition instructions."""
    print("=" * 70)
    print("    INDIAN CRICKET TEAM FACE DATASET - ACQUISITION GUIDE")
    print("=" * 70)
    print("\nRECOMMENDED KAGGLE DATASETS:")
    for i, d in enumerate(RECOMMENDED_DATASETS, 1):
        print(f"\n[{i}] {d['title']}")
        print(f"    Dataset ID : {d['id']}")
        print(f"    URL        : {d['url']}")
        print(f"    Overview   : {d['notes']}")

    print("\nHOW TO USE EXTERNAL DATASETS:")
    print("----------------------------------------------------------------------")
    print("METHOD 1: Kaggle CLI (Automated)")
    print("  1. Place your kaggle.json key in: C:\\Users\\<YourUser>\\.kaggle\\kaggle.json")
    print("  2. Run:")
    print("     python download_kaggle_dataset.py --download shilpeeprasad/indian-cricketers-image-dataset")
    print("\nMETHOD 2: Manual Browser Download (Kaggle or Google Dataset Search)")
    print("  1. Visit the Kaggle URL above and click 'Download (Archive.zip)'.")
    print("  2. Run:")
    print("     python download_kaggle_dataset.py --extract path/to/downloaded.zip")
    print("  3. Or manually unzip so player folders are in 'data/raw/<player_name>/'.")
    print("\nMETHOD 3: Google Dataset Search")
    print("  Search query: 'Indian cricketer face dataset' on https://datasetsearch.research.google.com/")
    print("  Download images and organize into: data/raw/<player_name>/")
    print("\nAFTER LOADING DATA:")
    print("  Run: python train.py")
    print("  Then launch: streamlit run app.py")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Dataset management tool for Indian Cricketer Face Identification")
    parser.add_argument("--download", type=str, help="Dataset slug on Kaggle (e.g. shilpeeprasad/indian-cricketers-image-dataset)")
    parser.add_argument("--extract", type=str, help="Path to local zip file to unpack into data/raw")
    parser.add_argument("--clean-names", action="store_true", help="Normalize and clean all player folder names in data/raw")
    parser.add_argument("--guide", action="store_true", help="Show Kaggle & Google Dataset Search instructions")

    args = parser.parse_args()

    if args.download:
        download_with_kaggle_api(args.download)
    elif args.extract:
        extract_zip(args.extract)
    elif args.clean_names:
        clean_raw_folder("data/raw")
    else:
        print_guide()


if __name__ == "__main__":
    main()

