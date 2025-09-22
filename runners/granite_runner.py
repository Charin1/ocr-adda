# runners/granite_runner.py

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

class GraniteRunner:
    """
    An OCR runner using the small and efficient IBM Granite-DocLing model.
    This model is well-suited for running on CPU or Apple Silicon MPS.
    """
    name = "granite-docling-258M"

    def __init__(self):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend is not available on this device.")
        
        self.device = torch.device("mps")
        model_id = "ibm-granite/granite-docling-258M"
        
        # Load the model and processor from Hugging Face
        # We load in default precision (float16/32) as it's small enough for 16GB RAM
        self.model = AutoModelForVision2Seq.from_pretrained(model_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # This is the specific prompt the model expects for OCR tasks
        self.task_prompt = "<doc_ocr_answer>"

    def run_image(self, img_path: str) -> str:
        """
        Processes a single image using the Granite-DocLing model.
        """
        image = Image.open(img_path).convert("RGB")
        
        try:
            # Prepare the inputs for the model
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate the text
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096, # Set a generous limit for dense pages
                prompt=self.task_prompt
            )
            
            # Decode the generated IDs to text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text.strip()

        except Exception as e:
            print(f"Error during Granite-DocLing inference: {e}")
            return f"[Granite-DocLing Error: {e}]"
        finally:
            # Clean up to prevent memory leaks
            del inputs
            torch.mps.empty_cache()