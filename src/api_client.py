import requests
import os
from PIL import Image
from io import BytesIO

class StableDiffusionAPIClient:
    """
    Клиент для вызова SD через Replicate API.
    Используется для тестирования, сравнения или препроцессинга.
    Не используется в основном пайплайне LoRA.
    """

    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN is required")

        self.headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json"
        }
        self.endpoint = "https://api.replicate.com/v1/models/runwayml/stable-diffusion-v1-5/predictions"

    def generate(self, prompt: str, negative_prompt: str = "", **kwargs) -> Image.Image:
        payload = {
            "input": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": kwargs.get("width", 512),
                "height": kwargs.get("height", 512),
                "num_inference_steps": kwargs.get("num_inference_steps", 30),
                "guidance_scale": kwargs.get("guidance_scale", 7.5),
                "seed": kwargs.get("seed", 42)
            }
        }

        response = requests.post(self.endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        prediction = response.json()

        while prediction["status"] not in ("succeeded", "failed"):
            import time
            time.sleep(2)
            status_url = prediction["urls"]["get"]
            response = requests.get(status_url, headers=self.headers)
            prediction = response.json()

        if prediction["status"] == "failed":
            raise RuntimeError(f"Prediction failed: {prediction.get('error')}")

        output_url = prediction["output"][0]
        img_data = requests.get(output_url).content
        return Image.open(BytesIO(img_data))