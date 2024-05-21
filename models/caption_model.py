from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
from utils.preprocess import preprocess_image

class CaptionModel:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate_caption(self, image_path: str) -> str:
        image = preprocess_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption