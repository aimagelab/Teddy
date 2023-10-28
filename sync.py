import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("src_path", type=Path)
parser.add_argument("dst_path", type=Path)
parser.add_argument("-f", "--force", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-e", "--exclude", type=str, nargs="*", default=[], help="exclude extensions")
args = parser.parse_args()

src_path = args.src_path
dst_path = args.dst_path

if not src_path.exists():
    raise ValueError(f"src_path {src_path} does not exist")
if not dst_path.exists():
    raise ValueError(f"dst_path {dst_path} does not exist")

for src_file in src_path.rglob("*"):
    dst_file = dst_path / src_file.relative_to(src_path)
    if src_file.is_dir():
        dst_file.mkdir(parents=True, exist_ok=True)
        continue
    if src_file.suffix in args.exclude:
        continue
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if not dst_path.exists() or args.force:
        try:
            shutil.copy(src_file, dst_file)
            if args.verbose:
                print(f"copy {src_file} to {dst_file}")
        except Exception as e:
            print(f"failed to copy {src_file} to {dst_file}: {e}")
