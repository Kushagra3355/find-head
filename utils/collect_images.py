import os
import shutil
from pathlib import Path

# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", 
    ".tiff", ".tif", ".heic", ".gif", ".raw", 
    ".cr2", ".nef", ".arw", ".dng"
}

def get_unique_destination_path(dest_dir: Path, filename: str) -> Path:
    """
    Returns a unique file path in dest_dir to prevent overwriting existing files.
    If 'image.jpg' exists, returns 'image_1.jpg', 'image_2.jpg', etc.
    """
    dest_path = dest_dir / filename
    if not dest_path.exists():
        return dest_path

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1

    while True:
        new_filename = f"{stem}_{counter}{suffix}"
        new_dest_path = dest_dir / new_filename
        if not new_dest_path.exists():
            return new_dest_path
        counter += 1

def collect_images(
    source_directory: str = r"E:\AiProff-Rootinly\find head\Selfie-Image-Detection-Dataset",
    output_folder_name: str = "images_unmasked",
    target_directory: str = None
):
    source_dir = Path(source_directory).resolve()
    target_dir = Path(target_directory).resolve() if target_directory else (source_dir / output_folder_name)

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source Directory : {source_dir}")
    print(f"Target Directory : {target_dir}")
    print("-" * 60)

    total_found = 0
    total_copied = 0
    duplicates_renamed = 0

    # Walk through directory recursively
    for root, dirs, files in os.walk(source_dir):
        current_path = Path(root).resolve()

        # Skip the output directory itself
        if target_dir in current_path.parents or current_path == target_dir:
            continue

        for file in files:
            file_path = current_path / file
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                total_found += 1
                
                dest_file_path = get_unique_destination_path(target_dir, file)
                
                if dest_file_path.name != file:
                    duplicates_renamed += 1
                    print(f"[RENAME CONFLICT] {file_path.relative_to(source_dir)} -> {dest_file_path.name}")
                else:
                    print(f"[COPY] {file_path.relative_to(source_dir)} -> {dest_file_path.name}")

                shutil.copy2(file_path, dest_file_path)
                total_copied += 1

    print("-" * 60)
    print("Summary:")
    print(f"  Total images found   : {total_found}")
    print(f"  Total images copied  : {total_copied}")
    print(f"  Duplicates renamed   : {duplicates_renamed}")
    print(f"  Output folder        : {target_dir}")

if __name__ == "__main__":
    # Current script directory as base (E:\AiProff-Rootinly\find head)
    current_dir = Path(__file__).parent.resolve()
    source_dir = current_dir / "Selfie-Image-Detection-Dataset"
    target_dir = current_dir / "invalid"
    
    collect_images(source_directory=source_dir, target_directory=target_dir)
