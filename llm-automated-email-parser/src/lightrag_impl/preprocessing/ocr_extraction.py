import os
import base64
import io
from typing import List
from openai import OpenAI
from pdf2image import convert_from_path # Requires: pip install pdf2image
from PIL import Image

class OCR_LLMPredictor:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OCR_LLM_BINDING_HOST", "http://localhost:8004/v1"),
            api_key=os.getenv("OCR_LLM_BINDING_API_KEY", "EMPTY"),
            max_retries=3
        )
        self.model_name = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-OCR")

    def _convert_pdf_to_base64_images(self, pdf_path: str) -> List[str]:
        """Converts PDF pages to base64 encoded strings."""
        # Convert PDF to a list of PIL Images
        images = convert_from_path(pdf_path)
        base64_strings = []
        
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            base64_strings.append(base64.encodebytes(buffered.getvalue()).decode('utf-8'))
            
        return base64_strings

    def __call__(self, pdf_path: str, prompt: str) -> List[str]:
        """
        Converts the PDF to images and processes them similar to the vLLM example.
        """
        # 1. Convert PDF to images (base64)
        base64_images = self._convert_pdf_to_base64_images(pdf_path)
        
        responses = []
        
        # 2. Process each page (DeepSeek-OCR processes images)
        for b64_img in base64_images:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"<image>\n{prompt}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                            }
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=8192
            )
            responses.append(response.choices[0].message.content)
            
        return responses