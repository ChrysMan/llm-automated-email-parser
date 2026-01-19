import os, io, base64, sys, json
from time import time
from typing import List
from openai import OpenAI
from pdf2image import convert_from_path # Requires: pip install pdf2image
import tempfile
from PIL import Image
from utils.logging import LOGGER
from utils.file_io import write_file

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
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
        images = convert_from_path(pdf_path, dpi=300)
        base64_strings = []
        
        for img in images:
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        
            buffered = io.BytesIO()
            # Save as PNG to avoid JPEG artifacts
            img.save(buffered, format="PNG")
            base64_strings.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            
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
                            {"type": "text", "text": f"<image>\n<|grounding|>{prompt}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                            }
                        ]
                    }
                ],
                extra_body={
                    "vllm_xargs": {
                        "ngram_size": 30,
                        "window_size": 90,
                        "whitelist_token_ids":[128821, 128822],
                    },
                    "skip_special_tokens": False
                },
                #response_format={ "type": "json_object" },
                temperature=0.0,
                max_tokens=2048
            )
            responses.append(response.choices[0].message.content)
            
        return responses
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS" # Most stable for V100
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

MODEL_INSTANCE = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        dtype='float16',
        enforce_eager=True,
        # trust_remote_code=True,
        gpu_memory_utilization=0.6
        # #max_num_seqs=5,
        # enable_prefix_caching=False,
        # mm_processor_cache_gb=0,
        # logits_processors=[NGramPerReqLogitsProcessor],      
    )

def llm(file_path: str, prompt: str):
    sampling_param = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                # ngram logit processor args
                extra_args=dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
                ),
                skip_special_tokens=False,
    )

    # Convert PDF to images in memory (much faster)
    try:
        with tempfile.TemporaryDirectory() as path:
            images_from_path = convert_from_path(file_path, output_folder=path)
            
            # Format the prompt correctly for DeepSeek-OCR
            formatted_prompt = f"<image>\n<|grounding|>{prompt}"

            final_text = ""
            for i, pil_img in enumerate(images_from_path):
                # Convert to RGB to ensure compatibility
                rgb_img = pil_img.convert("RGB")
                
                model_input = {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"image": rgb_img}
                }

                # Generate output
                model_outputs = MODEL_INSTANCE.generate(model_input, sampling_param)

                # Print output
                for output in model_outputs:
                    final_text += output.outputs[0].text + "\n"
                return final_text

    except Exception as e:
        return f"PDF Error: {str(e)}"

    
def main():
    tic1 = time()
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python ocr_extraction.py <dir_path>")
        sys.exit(1)

    dir_path = sys.argv[1]
    if not os.path.isdir(dir_path):
        LOGGER.error(f"{dir_path} is not a valid directory.")
        sys.exit(1)

    #predictor = OCR_LLMPredictor()

    for filename in os.listdir(dir_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(dir_path, filename)
            output_path = file_path.replace(".pdf", ".txt")
            try:
                prompt = "Extract all text from this document accurately. Output the result as a JSON object with a 'text' field containing the full transcription. If text is in Greek convert it to English"
                result = llm(file_path, prompt)
                
                write_file(result, output_path)
                # with open(output_path, "w", encoding="utf-8") as file:
                #     json.dump(result, file, indent=4, ensure_ascii=False, default=str)
            except Exception as e:
                LOGGER.error(f"Failed to extract or clean email from {filename}: {e}")
                continue

    LOGGER.info(f"Time taken to process: {time() - tic1} seconds")


if __name__ == "__main__":
    main()