#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


SOURCE_DIR = Path("tools") / "amazon-reviews" / "processed" / "Arts_Crafts_and_Sewing__raw_5core"
TARGET_DIR = Path("data") / "datasets" / "amazon-reviews"

REQUIRED_FILES = [
    "products_filtered_with_embeddings.parquet",
    "reviews_filtered_with_embeddings.parquet",
]

REQUIRED_DIRS = [
    "images",
]


def copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing required file: '{source}'.\nSee README.md for instructions to prepare the dataset.")

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        target.unlink()

    shutil.copy2(source, target)
    print(f"Copied '{source.name}'")


def copy_dir(source: Path, target: Path) -> None:
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Missing required directory: '{source}'.\nSee README.md for instructions to prepare the dataset.")

    if target.exists():
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)

    print(f"Copied '{source.name}'")


def main() -> int:
    print(f"Source: '{SOURCE_DIR}'")
    print(f"Target: '{TARGET_DIR}'")

    try:
        for filename in REQUIRED_FILES:
            copy_file(
                source=SOURCE_DIR / filename,
                target=TARGET_DIR / filename,
            )

        for dirname in REQUIRED_DIRS:
            copy_dir(
                source=SOURCE_DIR / dirname,
                target=TARGET_DIR / dirname,
            )

    except Exception as exc:
        print(f"Error: '{exc}'")
        return 1

    print(f"Runtime Amazon Reviews dataset copied to '{TARGET_DIR}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())