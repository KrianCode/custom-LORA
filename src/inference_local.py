# src/inference_local.py

from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch
from pathlib import Path

class LoRALocalInference:
    _instance = None

    def __new__(cls, base_model: str, device: str = "cuda"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_model: str, device: str = "cuda"):
        if self._initialized:
            return
        self.device = device
        self.base_model = base_model
        self.pipeline = None
        self.current_lora = None
        self.load_base_model()
        self._initialized = True

    def load_base_model(self):
        print(f"Loading base model: {self.base_model}")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)

        if self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()  # экономия VRAM

        self.text_encoder = self.pipeline.text_encoder
        self.vae = self.pipeline.vae
        self.unet_base = self.pipeline.unet  # сохраняем оригинальный UNet

    def load_lora(self, lora_path: str):
        """Загружает LoRA адаптер в текущий UNet"""
        if self.current_lora == lora_path:
            return  # уже загружен

        # Восстанавливаем оригинальный UNet
        self.pipeline.unet = self.unet_base

        # Загружаем LoRA
        self.pipeline.unet = PeftModel.from_pretrained(self.pipeline.unet, lora_path)
        self.pipeline.unet.to(self.device).eval()
        self.current_lora = lora_path
        print(f"LoRA loaded: {lora_path}")

    def generate(self, prompt: str, lora_path: str, negative_prompt: str = "", **kwargs):
        self.load_lora(lora_path)

        generator = torch.Generator(device=self.device).manual_seed(kwargs.get("seed", 42))

        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=kwargs.get("num_inference_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            width=kwargs.get("width", 512),
            height=kwargs.get("height", 512),
            generator=generator
        ).images[0]

        return image