# app.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import uuid
import os
from src.inference_local import LoRALocalInference
import torch

app = FastAPI()

# Инициализируем генератор один раз
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = LoRALocalInference("runwayml/stable-diffusion-v1-5", device=device)

@app.post("/upload_style")
async def upload_style(files: list[UploadFile]):
    upload_dir = "./data/user_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    for file in files:
        ext = file.filename.split('.')[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        with open(f"{upload_dir}/{filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"status": "uploaded", "count": len(files)}

@app.post("/train_style")
async def train_style(background_tasks: BackgroundTasks):
    # Асинхронно запускаем обучение
    background_tasks.add_task(run_training)
    return {"status": "training started"}

def run_training():
    import subprocess
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Training failed:", result.stderr)
    else:
        print("✅ Training finished")

@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    lora_name: str = Form("final"),
    seed: int = Form(42)
):
    lora_path = f"./models/loras/lora_{lora_name}"
    if not os.path.exists(lora_path):
        return {"error": f"LoRA {lora_name} not found"}

    image = generator.generate(
        prompt=prompt,
        lora_path=lora_path,
        seed=seed
    )

    output_path = f"./outputs/{uuid.uuid4()}.png"
    os.makedirs("./outputs", exist_ok=True)
    image.save(output_path)

    return FileResponse(output_path, media_type="image/png")