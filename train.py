from src.data_preprocess import DataPreprocessor
from src.train_lora import LoRATrainer
from transformers import CLIPTokenizer
from datasets import Dataset
import torch

def tokenize_captions(examples, tokenizer, max_length=77):
    captions = list(examples["caption"])
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    return {"input_ids": text_inputs.input_ids}

def main():
    # 1. Препроцессинг
    preprocessor = DataPreprocessor(image_dir="./data/user_uploads", target_size=(512, 512))
    dataset = preprocessor.load_and_preprocess_images()

    # 2. Токенизация
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = dataset.map(
        lambda x: tokenize_captions(x, tokenizer),
        batched=True,
        remove_columns=["caption"]
    )
    dataset.set_format(type="torch", columns=["image", "input_ids"])

    # 3. Обучение
    trainer = LoRATrainer(config_path="configs/default.yaml")
    trainer.train(dataset, num_epochs=100)

if __name__ == "__main__":
    main()