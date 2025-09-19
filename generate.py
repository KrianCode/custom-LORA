from src.inference_local import LoRALocalInference
import torch
def main():
    base_model = "runwayml/stable-diffusion-v1-5"
    lora_path = "./models/loras/lora_final"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = LoRALocalInference(base_model, device=device)
    prompt = "a cat sitting on a windowsill, in user-defined style"
    image = generator.generate(prompt, lora_path=lora_path, seed=1234)
    image.save("output_with_lora.png")
    print("✅ Generated with LoRA and saved to output_with_lora.png")
if __name__ == "__main__":
    main()