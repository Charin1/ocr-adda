# runners/api_runner.py

import os
from google import genai
from PIL import Image

class GeminiRunner:
    """
    An OCR runner that uses the Google Gen AI SDK (google-genai),
    consistent with the create_ground_truth script.
    """
    name = "models/gemini-2.5-flash"

    def __init__(self):
        # Use the explicit Client pattern for consistency.
        # The client will automatically look for the GOOGLE_API_KEY environment variable.
        try:
            self.client = genai.Client()
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
        except Exception as e:
            # Raise a runtime error to be caught by main.py's initialization loop
            raise RuntimeError(f"Failed to initialize Google Gen AI Client: {e}") from e

        # Define the model name and the prompt for the OCR task.
        self.model_name = 'models/gemini-2.5-flash'
        self.prompt = "Perform OCR on this document image. Extract all text content accurately, preserving the original line breaks and structure as much as possible."

    def run_image(self, img_path: str) -> str:
        """
        Processes a single image using the google-genai SDK's non-streaming method.
        """
        try:
            img = Image.open(img_path)
            
            # Use the client.models.generate_content pattern, consistent with the other script.
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.prompt, img] # Multimodal input
            )
            
            return response.text.strip()

        except Exception as e:
            # The error will be caught and reported by the main loop in main.py
            print(f"Error processing image with Gemini API: {e}")
            # Re-raise the exception so the main loop can handle it gracefully
            raise e