import argparse
import os
import random
import shutil
import time
from pathlib import Path

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tiff", ".tif", ".heic", ".gif"
}

def copy_random_images(
    source_dir: Path,
    dest_dir: Path,
    count: int = 5783,
    seed: int | None = None,
    dry_run: bool = False
) -> None:
    """
    Randomly selects `count` images from source_dir and copies them into dest_dir.
    """
    source_dir = Path(source_dir).resolve()
    dest_dir = Path(dest_dir).resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    print("=" * 60)
    print("Random Image Copier")
    print("=" * 60)
    print(f"Source Directory : {source_dir}")
    print(f"Target Directory : {dest_dir}")
    print(f"Requested Count  : {count}")
    if seed is not None:
        print(f"Random Seed      : {seed}")
    if dry_run:
        print("Mode             : DRY RUN (no files will be copied)")
    print("-" * 60)

    # 1. Scan for valid image files
    print("Scanning for image files...")
    image_files = [
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    total_found = len(image_files)
    print(f"Found {total_found} image(s) in source directory.")

    if total_found == 0:
        print("No matching image files found. Exiting.")
        return

    if count > total_found:
        print(
            f"Warning: Requested count ({count}) is greater than available images ({total_found}). "
            f"All {total_found} images will be copied."
        )
        count = total_found

    # 2. Randomly sample images
    if seed is not None:
        random.seed(seed)

    selected_files = random.sample(image_files, count)
    print(f"Randomly selected {len(selected_files)} images.")

    if dry_run:
        print("\nDry run completed. Sample of selected files:")
        for sample_file in selected_files[:5]:
            print(f"  - {sample_file.name}")
        if len(selected_files) > 5:
            print(f"  ... and {len(selected_files) - 5} more.")
        return

    # 3. Create target directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 4. Copy files with progress reporting
    print("\nCopying files...")
    start_time = time.time()
    copied_count = 0

    try:
        from tqdm import tqdm
        progress_iter = tqdm(selected_files, desc="Copying images", unit="file")
    except ImportError:
        progress_iter = selected_files

    for i, file_path in enumerate(progress_iter, start=1):
        target_path = dest_dir / file_path.name
        shutil.copy2(file_path, target_path)
        copied_count += 1

        # Fallback logging if tqdm is not installed
        if not hasattr(progress_iter, "update") and (i % 500 == 0 or i == count):
            print(f"  Copied {i}/{count} files...")

    elapsed_time = time.time() - start_time
    print("-" * 60)
    print("Summary:")
    print(f"  Total images available : {total_found}")
    print(f"  Successfully copied    : {copied_count}")
    print(f"  Destination directory  : {dest_dir}")
    print(f"  Time taken             : {elapsed_time:.2f} seconds")
    print("=" * 60)

def main():
    base_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(
        description="Randomly select and copy images from one folder to another."
    )
    parser.add_argument(
        "-s", "--source",
        type=Path,
        default=base_dir / "invalid",
        help="Path to the source folder containing images (default: 'invalid' in current directory)."
    )
    parser.add_argument(
        "-d", "--dest",
        type=Path,
        default=base_dir / "invalid_junk",
        help="Path to the destination folder (default: 'invalid_junk' in current directory)."
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=5783,
        help="Number of random images to copy (default: 5783)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a trial run without copying files."
    )

    args = parser.parse_args()

    copy_random_images(
        source_dir=args.source,
        dest_dir=args.dest,
        count=args.count,
        seed=args.seed,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
