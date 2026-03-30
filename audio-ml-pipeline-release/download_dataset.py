"""
DNS Challenge Subset Downloader
Downloads a training-ready subset of the DNS Challenge 4 dataset.
Total download: ~4-5GB (sufficient for U-Net training)

Clean speech selected:
- VocalSet_48kHz_mono: 974MB - professional singers, great vocal variety
- emotional_speech: 2.4GB - emotional speech patterns

Noise selected:
- freesound_000: ~1GB - diverse real-world noise types
"""

import urllib.request
import os
import tarfile
from tqdm import tqdm

AZURE_URL   = "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data")

DOWNLOADS = [
    {
        "blob": "clean_fullband/datasets_fullband.clean_fullband.VocalSet_48kHz_mono_000_NA_NA.tar.bz2",
        "description": "VocalSet clean speech (974MB)"
    },
    {
        "blob": "clean_fullband/datasets_fullband.clean_fullband.emotional_speech_000_NA_NA.tar.bz2",
        "description": "Emotional speech clean (2.4GB)"
    },
    {
        "blob": "noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2",
        "description": "Freesound noise pack (varies)"
    },
]

class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract(blob, description):
    url = f"{AZURE_URL}/{blob}"
    filename = blob.split("/")[-1]
    dest_dir = os.path.join(OUTPUT_PATH, os.path.dirname(blob))
    dest_path = os.path.join(OUTPUT_PATH, blob)

    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest_path):
        print(f"Already downloaded: {filename}")
    else:
        print(f"\nDownloading {description}...")
        print(f"URL: {url}")
        try:
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
            print(f"Download complete: {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False

    print(f"Extracting {filename}...")
    try:
        extract_path = os.path.join(OUTPUT_PATH, os.path.dirname(blob))
        with tarfile.open(dest_path, 'r:bz2') as tar:
            tar.extractall(extract_path)
        print(f"Extracted successfully.")
        os.remove(dest_path)
        print(f"Archive removed to save space.")
        return True
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        return False

print("=" * 60)
print("DNS Challenge Dataset Downloader")
print("=" * 60)
print(f"Output directory: {OUTPUT_PATH}")
print(f"Files to download: {len(DOWNLOADS)}")
print("=" * 60)

for item in DOWNLOADS:
    success = download_and_extract(item["blob"], item["description"])
    if success:
        print(f"Done: {item['description']}\n")
    else:
        print(f"Failed: {item['description']}\n")

print("\nAll downloads complete.")
print("Data ready for training in:", OUTPUT_PATH)
