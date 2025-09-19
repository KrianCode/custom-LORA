import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple
from torchvision import transforms
from datasets import Dataset

class DataPreprocessor:
    def __init__(self, image_dir: str, target_size: Tuple[int, int] = (512, 512)):
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def load_and_preprocess_images(self) -> Dataset:
        """
        Загружает изображения, ресайзит, нормализует, возвращает Dataset.
        """
        image_paths = [
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self.valid_extensions
        ]

        if not image_paths:
            raise ValueError(f"No valid images found in {self.image_dir}")

        images = []
        captions = []

        for img_path in image_paths:
            try:
                img = self._load_image(img_path)
                img = self._resize_and_center_crop(img)
                img = self._normalize_image(img)
                images.append(img)

                captions.append("a photo in user-defined style")

            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")

        dataset = Dataset.from_dict({
            "image": images,
            "caption": captions,
            "image_path": [str(p) for p in image_paths]
        })

        return dataset

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _resize_and_center_crop(self, img: Image.Image) -> Image.Image:
        # Сохраняем пропорции, делаем center crop
        w, h = img.size
        scale = max(self.target_size[0] / w, self.target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        left = (new_w - self.target_size[0]) // 2
        top = (new_h - self.target_size[1]) // 2
        right = left + self.target_size[0]
        bottom = top + self.target_size[1]

        img = img.crop((left, top, right, bottom))
        return img

    def _normalize_image(self, img: Image.Image) -> np.ndarray:
        # Нормализуем в [-1, 1] для SD
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - 0.5) * 2.0
        return img.transpose(2, 0, 1)  # HWC -> CHW