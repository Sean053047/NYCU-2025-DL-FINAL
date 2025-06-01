#!/usr/bin/env python3
# keep_checkpoint.py
import argparse, pathlib, re, shutil, sys

pat = re.compile(r"checkpoint-(\d+)$")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dir", help="checkpoint 資料夾路徑")
    pa.add_argument("--max", type=int, default=1, help="最多保留幾個")
    pa.add_argument("--dry-run", action="store_true", help="只顯示將被刪除的項目")
    args = pa.parse_args()

    root = pathlib.Path(args.dir)
    ckpts = [p for p in root.iterdir() if p.is_dir() and pat.fullmatch(p.name)]
    ckpts.sort(key=lambda p: int(p.name.split("-")[1]))  # 依 step 排序

    excess = len(ckpts) - args.max
    if excess > 0:
        for p in ckpts[:excess]:
            print("Removing", p)
            if not args.dry_run:
                shutil.rmtree(p, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())