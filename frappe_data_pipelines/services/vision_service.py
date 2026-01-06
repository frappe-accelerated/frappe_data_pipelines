"""
Vision Service for processing images and visual content.

Uses vision LLMs (via OpenRouter) to describe images, charts,
infographics, and other visual elements in documents.
"""
import frappe
import base64
import os
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VisualContent:
    """Processed visual content from an image."""
    ocr_text: str  # Text extracted via OCR
    description: str  # LLM-generated description
    combined: str  # Combined text for embedding
    source_path: str
    detected_elements: List[str] = None  # charts, tables, diagrams, etc.


class VisionService:
    """
    Process images and visual content using vision LLMs.

    Combines OCR for text extraction with vision LLM for
    semantic understanding of visual content.
    """

    IMAGE_DESCRIPTION_PROMPT = """Analyze this image from a document and provide a detailed description.

Include the following in your response:
1. **Visual Content**: What type of image is this? (chart, diagram, photo, infographic, table, etc.)
2. **Key Information**: What are the main data points, statistics, or facts shown?
3. **Text Content**: Transcribe any visible text, labels, or annotations.
4. **Context**: What is the main message or insight this image conveys?
5. **Relationships**: If applicable, describe any relationships, trends, or comparisons shown.

Format your response as structured text suitable for search indexing.
Be thorough but concise - aim for 200-400 words."""

    SUPPORTED_IMAGE_TYPES = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif'
    }

    def __init__(self):
        self.settings = frappe.get_single("Data Pipeline Settings")
        self.vision_model = self.settings.vision_model or "qwen/qwen3-vl-235b-a22b-instruct"
        self.api_key = self.settings.get_password("openrouter_api_key")

        if not self.api_key:
            frappe.throw("OpenRouter API key is required for vision processing")

    @classmethod
    def is_image(cls, file_path: str) -> bool:
        """Check if a file is a supported image type."""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_IMAGE_TYPES

    def process_image(self, image_path: str, ocr_text: str = "") -> VisualContent:
        """
        Process an image and generate structured content for embedding.

        Args:
            image_path: Path to the image file
            ocr_text: Optional pre-extracted OCR text

        Returns:
            VisualContent with description and combined text
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Get image description from vision LLM
        description = self._describe_with_vision_llm(image_path)

        # Combine OCR text and description for embedding
        combined_parts = []
        if description:
            combined_parts.append(f"Visual Description:\n{description}")
        if ocr_text:
            combined_parts.append(f"\nExtracted Text:\n{ocr_text}")

        combined = "\n".join(combined_parts) if combined_parts else ""

        # Detect visual element types from description
        detected_elements = self._detect_visual_elements(description)

        return VisualContent(
            ocr_text=ocr_text,
            description=description,
            combined=combined,
            source_path=image_path,
            detected_elements=detected_elements
        )

    def _describe_with_vision_llm(self, image_path: str) -> str:
        """
        Get description from vision LLM via OpenRouter.

        Args:
            image_path: Path to the image file

        Returns:
            Generated description text
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = f.read()

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Detect MIME type
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
        }
        mime_type = mime_types.get(ext, 'image/png')

        # Build the data URL
        image_url = f"data:{mime_type};base64,{image_b64}"

        return self._call_openrouter_vision(image_url, self.IMAGE_DESCRIPTION_PROMPT)

    def _call_openrouter_vision(self, image_url: str, prompt: str) -> str:
        """
        Call OpenRouter API with a vision model.

        Args:
            image_url: Base64 data URL of the image
            prompt: Text prompt for the vision model

        Returns:
            Generated text response
        """
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        response = client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()

        return ""

    def _detect_visual_elements(self, description: str) -> List[str]:
        """
        Detect types of visual elements mentioned in the description.

        Args:
            description: The LLM-generated description

        Returns:
            List of detected element types
        """
        elements = []
        description_lower = description.lower()

        element_keywords = {
            'chart': ['chart', 'graph', 'plot', 'histogram', 'pie chart', 'bar chart', 'line graph'],
            'table': ['table', 'grid', 'spreadsheet', 'matrix'],
            'diagram': ['diagram', 'flowchart', 'flow chart', 'schematic', 'architecture'],
            'infographic': ['infographic', 'info graphic', 'visual summary'],
            'photo': ['photo', 'photograph', 'picture', 'image'],
            'screenshot': ['screenshot', 'screen capture', 'screen shot'],
            'map': ['map', 'geographic', 'location'],
            'logo': ['logo', 'brand', 'icon'],
            'equation': ['equation', 'formula', 'mathematical'],
        }

        for element_type, keywords in element_keywords.items():
            if any(kw in description_lower for kw in keywords):
                elements.append(element_type)

        return elements if elements else ['image']

    def process_document_images(
        self,
        images: List[str],
        ocr_texts: Optional[List[str]] = None
    ) -> List[VisualContent]:
        """
        Process multiple images from a document.

        Args:
            images: List of image file paths
            ocr_texts: Optional list of OCR texts corresponding to images

        Returns:
            List of VisualContent objects
        """
        results = []

        for i, image_path in enumerate(images):
            ocr_text = ocr_texts[i] if ocr_texts and i < len(ocr_texts) else ""

            try:
                result = self.process_image(image_path, ocr_text)
                results.append(result)
            except Exception as e:
                frappe.log_error(
                    title=f"Vision processing failed for {image_path}",
                    message=str(e)
                )
                # Create a minimal result on failure
                results.append(VisualContent(
                    ocr_text=ocr_text,
                    description="",
                    combined=ocr_text,
                    source_path=image_path,
                    detected_elements=['image']
                ))

        return results

    def test_connection(self) -> dict:
        """Test connection to the vision model."""
        try:
            # Create a simple test image (1x1 white pixel PNG)
            test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
            test_url = f"data:image/png;base64,{test_image_b64}"

            result = self._call_openrouter_vision(test_url, "What do you see? Reply briefly.")

            if result:
                return {
                    "success": True,
                    "message": f"Vision model '{self.vision_model}' is working."
                }
            return {
                "success": False,
                "message": "No response from vision model"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Vision processing failed: {str(e)}"
            }


def get_vision_service() -> Optional[VisionService]:
    """
    Factory function to get vision service if enabled.

    Returns:
        VisionService if enabled, None otherwise
    """
    try:
        settings = frappe.get_single("Data Pipeline Settings")
        if settings.enable_smart_pipeline and settings.enable_vision_processing:
            return VisionService()
    except Exception:
        pass
    return None
