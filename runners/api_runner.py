# runners/api_runner.py

import os
from google import genai
from PIL import Image

class GeminiRunner:
    """
    A flexible OCR runner for any Google Gen AI model.
    The model name is now passed during initialization.
    """
    def __init__(self, api_model_name: str, display_name: str):
        """
        Initializes the runner with a specific API model name.

        Args:
            api_model_name: The exact model name to be called via the API (e.g., 'gemini-2.5-flash').
            display_name: The name to use for output folders and reports (e.g., 'gemini-2.5-flash').
        """
        # The display name is used for folders and reports
        self.name = display_name
        
        try:
            self.client = genai.Client()
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Gen AI Client: {e}") from e

        # The API model name is used for the actual API call
        self.model_name = api_model_name
        self.prompt = "Perform OCR on this document image. Extract all text content accurately, preserving the original line breaks and structure as much as possible."

    def run_image(self, img_path: str) -> str:
        try:
            img = Image.open(img_path)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.prompt, img]
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error processing image with Gemini API: {e}")
            raise e