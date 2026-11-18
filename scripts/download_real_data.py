"""Download official real-data releases without changing the seed datasets."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from typing import Dict, Iterable, Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROJECT_ROOT, ensure_dir  # noqa: E402
from src.data.schema import write_json, write_jsonl  # noqa: E402

EMOMETA_ZIP = "https://github.com/DUTIR-YSQ/EmoMeta/archive/refs/heads/master.zip"
TAOBOO_ROOT = "https://hf-mirror.com/datasets/TaoBao-MM/Taobao-MM/resolve/main/"
TAOBOO_FILES = {
    "raw_item_features.parquet": "raw/item_features.parquet",
    "train_train-shard-000159.parquet": "train/train-shard-000159.parquet",
    "train_metadata.json": "train/metadata.json",
    "test_test-shard-000048.parquet": "test/test-shard-000048.parquet",
    "test_metadata.json": "test/metadata.json",
}


def download(url: str, path: str, timeout: int = 120) -> Dict:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return {"path": path, "bytes": os.path.getsize(path), "status": "existing", "url": url}
    ensure_dir(os.path.dirname(path))
    temporary = path + ".part"
    with requests.get(url, stream=True, timeout=(30, timeout), headers={"User-Agent": "metonymyrec/1.0"}) as response:
        response.raise_for_status()
        with open(temporary, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    os.replace(temporary, path)
    return {"path": path, "bytes": os.path.getsize(path), "status": "downloaded", "url": url}


def prepare_emometa(raw_root: str, download_images: bool = False, image_limit: int = 0) -> Dict:
    directory = os.path.join(raw_root, "emometa")
    repository = os.path.join(directory, "EmoMeta-master")
    workbook = os.path.join(repository, "DataAll.xlsx")
    if not os.path.exists(workbook):
        archive = download(EMOMETA_ZIP, os.path.join(directory, "EmoMeta-master.zip"))
        with zipfile.ZipFile(archive["path"]) as bundle:
            bundle.extractall(directory)
    if not os.path.exists(workbook):
        raise FileNotFoundError("EmoMeta DataAll.xlsx was not found after extraction: " + workbook)

    result = {"repository": repository, "workbook": workbook, "status": "ready"}
    if download_images:
        import pandas as pd
        from PIL import Image

        frame = pd.read_excel(workbook)
        image_dir = ensure_dir(os.path.join(repository, "dataF"))
        failures = []
        downloaded = 0
        attempted = 0
        for _, row in frame.iterrows():
            if image_limit and attempted >= image_limit:
                break
            name = str(row["Img_id"])
            url = row.get("Img_url")
            destination = os.path.join(image_dir, name)
            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                continue
            attempted += 1
            if not isinstance(url, str) or not url.strip():
                failures.append({"image": name, "error": "official Img_url is empty"})
                continue
            temporary = destination + ".part"
            try:
                with requests.get(url, stream=True, timeout=(15, 35), headers={"User-Agent": "Mozilla/5.0"}) as response:
                    response.raise_for_status()
                    with open(temporary, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                handle.write(chunk)
                with Image.open(temporary) as image:
                    image.verify()
                os.replace(temporary, destination)
                downloaded += 1
            except Exception as error:
                if os.path.exists(temporary):
                    os.remove(temporary)
                failures.append({"image": name, "url": url, "error": str(error)})
        failure_path = os.path.join(directory, "image_download_failures.jsonl")
        write_jsonl(failure_path, failures)
        result["images"] = {
            "attempted": attempted,
            "downloaded": downloaded,
            "failed": len(failures),
            "failure_list": failure_path,
        }
    return result


def prepare_taobao(raw_root: str, with_keys: bool = False, with_full_embeddings: bool = False) -> Dict:
    directory = os.path.join(raw_root, "taobao_mm")
    files = dict(TAOBOO_FILES)
    if with_keys:
        files[os.path.join("feature_map", "scl_emb_int8_p90_keys.npy")] = "feature_map/scl_emb_int8_p90_keys.npy"
    if with_full_embeddings:
        files[os.path.join("feature_map", "scl_emb_int8_p90_values.npy")] = "feature_map/scl_emb_int8_p90_values.npy"
        files[os.path.join("raw", "scl_embedding_int8_p90.parquet")] = "raw/scl_embedding_int8_p90.parquet"
    result = {}
    for local_name, remote_name in files.items():
        result[local_name] = download(TAOBOO_ROOT + remote_name, os.path.join(directory, local_name))
    return result


def check_yelp(raw_root: str) -> Dict:
    directory = os.path.join(raw_root, "yelp")
    candidates = [
        "yelp_dataset.tar", "yelp_dataset.tar.gz", "photos.tar", "photos.tar.gz",
        "yelp_academic_dataset_photo.json", "yelp_academic_dataset_business.json",
        "yelp_academic_dataset_review.json",
    ]
    found = [os.path.join(directory, name) for name in candidates if os.path.exists(os.path.join(directory, name))]
    return {
        "status": "local_files_found" if found else "manual_download_required",
        "directory": directory,
        "files": found,
        "official_url": "https://www.yelp.com/dataset",
        "note": "Accept Yelp's licence and place the official archive or extracted JSON files here; no substitute data are generated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "emometa", "taobao", "yelp"], default="all")
    parser.add_argument("--raw-dir", default=os.path.join(PROJECT_ROOT, "data", "raw"))
    parser.add_argument("--images", action="store_true", help="try missing EmoMeta Img_url entries; dead URLs are logged")
    parser.add_argument("--image-limit", type=int, default=0, help="0 tries every missing image")
    parser.add_argument("--taobao-keys", action="store_true", help="download the sorted 283 MB SCL key index")
    parser.add_argument("--taobao-full-embeddings", action="store_true", help="download the full multi-GB SCL release")
    args = parser.parse_args()

    manifest = {}
    if args.dataset in ("all", "emometa"):
        manifest["emometa"] = prepare_emometa(args.raw_dir, args.images, args.image_limit)
    if args.dataset in ("all", "taobao"):
        manifest["taobao_mm"] = prepare_taobao(args.raw_dir, args.taobao_keys, args.taobao_full_embeddings)
    if args.dataset in ("all", "yelp"):
        manifest["yelp"] = check_yelp(args.raw_dir)
    path = os.path.join(ensure_dir(args.raw_dir), "download_manifest.json")
    write_json(path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("[download_real_data] manifest: " + path)


if __name__ == "__main__":
    main()
