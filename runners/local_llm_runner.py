# runners/local_llm_runner.py

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

class FlorenceRunner:
    """
    An OCR runner using a local, quantized version of Microsoft's Florence-2 model.
    """
    name = "florence2_base"

    def __init__(self):
        # Check for MPS availability on Apple Silicon
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend is not available on this device.")
        
        self.device = torch.device("mps")
        
        # Use a smaller, powerful model. 4-bit quantization is crucial for 16GB RAM.
        model_id = 'microsoft/Florence-2-base'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # Define the task prompt for OCR
        self.prompt = "<OCR>"

    def run_image(self, img_path: str) -> str:
        image = Image.open(img_path)

        # The `transformers` pipeline for Florence-2 is memory intensive.
        # Process one image at a time and clear memory.
        try:
            # Generate text based on the image and prompt
            inputs = self.processor(text=self.prompt, images=image, return_tensors="pt")
            
            # Move inputs to the Metal Performance Shaders (MPS) device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=2048,  # Adjust based on expected text length
                num_beams=3
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # The model output includes the prompt, so we parse the result
            parsed_answer = self.processor.post_process_generation(generated_text, task=self.prompt, image_size=image.size)
            
            return parsed_answer.get(self.prompt, "[Florence OCR parsing failed]")

        except Exception as e:
            print(f"Error during Florence-2 inference: {e}")
            return f"[Florence-2 Error: {e}]"
        finally:
            # Clean up to prevent memory leaks
            del inputs
            torch.mps.empty_cache()