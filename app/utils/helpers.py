import os
import shutil
from typing import List

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def clear_directory(path: str):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.is_dir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def get_all_files(directory: str, extensions: List[str] = [".pdf", ".txt"]) -> List[str]:
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files
