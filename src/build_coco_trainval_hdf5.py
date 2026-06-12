#!/usr/bin/env python
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Build trainval COCO 224x224 float16 HDF5 for adapter training.")
    parser.add_argument("--coco_root", default=os.environ.get("COCO_ROOT", "data/coco"))
    parser.add_argument("--out", default=os.environ.get("COCO_HDF5", "data/coco_images_224_float16.hdf5"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--compression", default=None, choices=[None, "gzip", "lzf"])
    parser.add_argument("--max_images", type=int, default=0)
    return parser.parse_args()


def center_crop_resize(path, size):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((size, size), Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return arr.astype(np.float16)


def load_batch(batch_paths, size, num_workers):
    if num_workers <= 1:
        return np.stack([center_crop_resize(path, size) for path in batch_paths], axis=0)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(lambda p: center_crop_resize(p, size), batch_paths))
    return np.stack(images, axis=0)


def main():
    args = parse_args()
    coco_root = Path(args.coco_root)
    image_paths = sorted((coco_root / "train2017").glob("*.jpg")) + sorted((coco_root / "val2017").glob("*.jpg"))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No jpg images found under {coco_root}/train2017 and val2017")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    n = len(image_paths)
    with h5py.File(tmp, "w") as f:
        images = f.create_dataset(
            "images",
            shape=(n, 3, args.size, args.size),
            dtype="float16",
            compression=args.compression,
        )

        for start in tqdm(range(0, n, args.chunk_size), desc="build_hdf5"):
            batch_paths = image_paths[start : start + args.chunk_size]
            batch = load_batch(batch_paths, args.size, args.num_workers)
            images[start : start + len(batch_paths)] = batch

        string_dtype = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("file_names", data=[p.name for p in image_paths], dtype=string_dtype)
        f.create_dataset("split", data=[p.parent.name for p in image_paths], dtype=string_dtype)

        meta = {
            "coco_root": str(coco_root),
            "num_images": n,
            "size": args.size,
            "compression": args.compression,
            "num_workers": args.num_workers,
            "train2017": sum(1 for p in image_paths if p.parent.name == "train2017"),
            "val2017": sum(1 for p in image_paths if p.parent.name == "val2017"),
        }
        f.attrs["metadata"] = json.dumps(meta)

    tmp.rename(out)
    print(json.dumps({"saved": str(out), "num_images": n}, indent=2))


if __name__ == "__main__":
    main()
