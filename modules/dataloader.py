import glob
import os
import random
import tarfile
import shutil
import hashlib
from PIL import Image, ImageOps
import numpy as np

import requests

from config import WINDOW_SIZE
from modules.integral_image import to_integral


def gamma(
    values: np.ndarray, coeff: float = 2.2
) -> np.ndarray:
    return values ** (1.0 / coeff)


def gleam(values: np.ndarray) -> np.ndarray:
    return (
        np.sum(gamma(values), axis=2)
        / values.shape[2]
    )


def to_float_array(
    img: Image.Image,
) -> np.ndarray:
    return (
        np.array(img).astype(np.float32) / 255.0
    )


def to_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.uint8(values * 255.0)
    )


def download_file(url: str, path: str):
    print("Downloading file ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    print("Download completed.")


def md5(
    path: str, chunk_size: int = 65536
) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(chunk_size), b""
        ):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def untar(file_path: str, dest_path: str):
    print("Extracting file.")
    with tarfile.open(file_path, "r:gz") as f:
        f.extractall(dest_path)
    print("Extraction completed.")


def open_face(
    path: str, resize: bool = True
) -> Image.Image:
    CROP_TOP = 50

    img = Image.open(path)
    img = to_image(
        gamma(to_float_array(img)[CROP_TOP:, :])
    )
    min_size = np.min(img.size)
    img = ImageOps.fit(
        img, (min_size, min_size), Image.ANTIALIAS
    )
    if resize:
        img = img.resize(
            (WINDOW_SIZE, WINDOW_SIZE),
            Image.ANTIALIAS,
        )
    return img.convert("L")


def random_crop(img: Image.Image,) -> Image.Image:
    max_allowed_size = np.min(img.size)
    size = random.randint(
        WINDOW_SIZE, max_allowed_size
    )
    max_width = img.size[0] - size - 1
    max_height = img.size[1] - size - 1
    left = (
        0
        if (max_width <= 1)
        else random.randint(0, max_width)
    )
    top = (
        0
        if (max_height <= 1)
        else random.randint(0, max_height)
    )
    return img.crop(
        (left, top, left + size, top + size,)
    )


def open_background(
    path: str, resize: bool = True
) -> Image.Image:
    img = Image.open(path)
    img = to_image(gleam(to_float_array(img)))
    img = random_crop(img)
    if resize:
        img = img.resize(
            (WINDOW_SIZE, WINDOW_SIZE),
            Image.ANTIALIAS,
        )
    return img.convert("L")


class Dataloader:
    def __init__(
        self, dataset_path: str = "datasets/set0",
    ) -> None:
        self.dataset_path = dataset_path
        self.faces_dir = os.path.join(
            dataset_path,
            "faces_aligned_small_mirrored_co_aligned_cropped_cleaned",
        )
        self.backgrounds_dir = os.path.join(
            dataset_path, "iccv09Data"
        )

    def load(self, p: int, n: int):
        """返回样本积分图数组及其对应标签列表

        Args:
            p (int): 正样本数
            n (int): 负样本数

        """        
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        face_image_files = self.load_faces()
        background_image_files = (
            self.load_background()
        )
        xs = []
        xs.extend(
            [
                to_integral(to_float_array(open_face(f)))
                for f in random.sample(
                    face_image_files, p
                )
            ]
        )
        xs.extend(
            [
                to_integral(to_float_array(open_background(f)))
                for f in np.random.choice(
                    background_image_files,
                    n,
                    replace=True,
                )
            ]
        )

        ys = np.hstack(
            [np.ones((p,)), np.zeros((n,))]
        )
        return np.array(xs), ys

    def load_faces(self) -> list:
        dataset_path = self.dataset_path
        faces_url = "https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=1"
        faces_md5 = (
            "ab853c17ca6630c191457ff1fb16c1a4"
        )

        faces_archive = os.path.join(
            dataset_path,
            "faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz",
        )
        faces_dir = self.faces_dir

        if (
            not os.path.exists(faces_archive)
            or md5(faces_archive) != faces_md5
        ):
            download_file(
                faces_url, faces_archive
            )

        if not os.path.exists(faces_dir):
            untar(faces_archive, dataset_path)

        return glob.glob(
            os.path.join(
                faces_dir, "**", "*.png"
            ),
            recursive=True,
        )

    def load_background(self) -> list:
        dataset_path = self.dataset_path
        backgrounds_url = "http://dags.stanford.edu/data/iccv09Data.tar.gz"
        backgrounds_md5 = (
            "f469cf0ab459d94990edcf756694f4d5"
        )

        backgrounds_archive = os.path.join(
            dataset_path, "iccv09Data.tar.gz"
        )
        backgrounds_dir = self.backgrounds_dir

        if (
            not os.path.exists(
                backgrounds_archive
            )
            or md5(backgrounds_archive)
            != backgrounds_md5
        ):
            download_file(
                backgrounds_url,
                backgrounds_archive,
            )

        if not os.path.exists(backgrounds_dir):
            untar(
                backgrounds_archive, dataset_path
            )

        return glob.glob(
            os.path.join(
                backgrounds_dir, "**", "*.jpg"
            ),
            recursive=True,
        )

