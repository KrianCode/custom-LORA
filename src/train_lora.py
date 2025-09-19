import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional
import yaml
import os
from pathlib import Path

class LoRATrainer:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.accelerator = Accelerator(
            mixed_precision=self.config.get('mixed_precision', 'fp16'),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            log_with="tensorboard",
            project_dir=self.config.get('output_dir', './logs')
        )

        self.setup_model()
        self.setup_optimizer()


    def setup_model(self):
        model_id = self.config['model_id']
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

        # Применяем LoRA только к attention слоям
        lora_config = LoraConfig(
            r=self.config.get('lora_rank', 8),
            lora_alpha=self.config.get('lora_alpha', 16),
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=self.config.get('lora_dropout', 0.0),
            bias="none"
        )

        self.unet = get_peft_model(unet, lora_config)
        self.unet.print_trainable_parameters()

        # Загружаем остальное
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            unet=self.unet,
            torch_dtype=torch.float16 if self.accelerator.mixed_precision == "fp16" else torch.float32
        ).to(self.accelerator.device)

        self.text_encoder = self.pipeline.text_encoder
        self.vae = self.pipeline.vae
        self.noise_scheduler = self.pipeline.scheduler

        # Замораживаем всё, кроме LoRA
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        for name, param in self.unet.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    def setup_optimizer(self):
        params_to_optimize = [p for p in self.unet.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-2)
        )

    def train(self, dataset, num_epochs: int = 100):
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        dataloader, self.unet, self.optimizer = self.accelerator.prepare(
            dataloader, self.unet, self.optimizer
        )

        global_step = 0
        for epoch in range(num_epochs):
            self.unet.train()
            for batch in dataloader:
                with self.accelerator.accumulate(self.unet):
                    latents = self.vae.encode(batch["image"].to(dtype=torch.float16)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                global_step += 1
                if global_step % self.config.get('log_every', 10) == 0:
                    self.accelerator.log({"loss": loss.item()}, step=global_step)

            if epoch % self.config.get('save_every_epoch', 10) == 0:
                self.save_lora_adapter(epoch)

        self.save_lora_adapter("final")

    def save_lora_adapter(self, name: str):
        output_dir = Path(self.config['output_dir']) / "loras"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(output_dir / f"lora_{name}")
        print(f"LoRA saved to {output_dir / f'lora_{name}'}")