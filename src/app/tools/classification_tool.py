import json
import base64
import logging
from pathlib import Path
from typing import List, Type, Dict, Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False

class ClassifyDocumentsInput(BaseModel):
    employee_id: str = Field(..., description="Employee identifier whose documents are in data_store/{employee_id}.")


class ClassifyDocumentsTool(BaseTool):
    name: str = "classify_documents"
    description: str = (
        "Classify all document images in data_store/{employee_id} into types: "
        "AADHAR_CARD, PAN_CARD, MARKSHEET, DEGREE_CERTIFICATE, EXPERIENCE_CERTIFICATE, "
        "SERVICE_LETTER, APPLICATION_FORM, PHOTO, AFFIDAVIT, RESUME, OTHER. "
        "Returns JSON mapping filenames to document types."
    )
    args_schema: Type[BaseModel] = ClassifyDocumentsInput

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 with resizing if PIL is available."""
        import os
        from io import BytesIO
        
        max_dim = 1024
        quality = 75

        try:
            if _HAS_PIL and Image is not None:
                with Image.open(image_path) as img:
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")

                    w, h = img.size
                    longest = max(w, h)
                    if longest > max_dim:
                        scale = max_dim / float(longest)
                        new_size = (int(w * scale), int(h * scale))
                        img = img.resize(new_size, Image.LANCZOS)

                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=quality, optimize=True)
                    buf.seek(0)
                    return base64.b64encode(buf.read()).decode('utf-8')
            else:
                with image_path.open("rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.exception("Failed to encode/resize image %s: %s", image_path, e)
            with image_path.open("rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')

    def _run(self, employee_id: str) -> str:
        """Classify all document images using vision AI."""
        streamer = getattr(self, '_streamer', None)
        employee_id = employee_id.strip().rstrip('}').rstrip('"').strip()
        
        try:
            root = Path(__file__).resolve().parents[3]
        except Exception:
            root = Path.cwd()
        
        data_store_dir = root / "data_store" / employee_id
        
        if not data_store_dir.exists() or not data_store_dir.is_dir():
            error_msg = f"Data store not found for employee {employee_id}"
            logger.error("[ClassifyTool] %s", error_msg)
            return json.dumps({"error": error_msg})
        
        images = sorted([
            f for f in data_store_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        
        if not images:
            error_msg = "No images found for classification"
            logger.error("[ClassifyTool] %s", error_msg)
            return json.dumps({"error": error_msg})

        logger.info("[ClassifyTool] Starting classification for employee: %s", employee_id)
        logger.info("[ClassifyTool] Found %d images to classify", len(images))
        
        if streamer:
            streamer.send_update("classification_start", {
                "agent": "classification",
                "tool": "ClassifyDocumentsTool",
                "message": "We are identifying each document type. Getting started…",
                "status": "Starting document classification…",
                "total_images": len(images)
            })
        
        import os
        from openai import OpenAI
        
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        model = os.getenv("VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        
        if not api_key:
            return json.dumps({"error": "LLM_API_KEY not set"})
        
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        
        classification_prompt = """You are an expert document classifier specializing in Indian identity and official documents.

        Analyze the given image carefully and classify it into EXACTLY ONE of the following categories:

        1. AADHAR_CARD — Indian Aadhaar card (shows 12-digit number, “UIDAI”, “Government of India”, or QR code).
        2. PAN_CARD — PAN card (shows 10-character alphanumeric code, “Income Tax Department”, or “Permanent Account Number”).
        3. MARKSHEET — Academic marksheet or transcript (shows subjects, marks, or grades).
        4. DEGREE_CERTIFICATE — Degree/diploma certificate (mentions “Degree”, “Diploma”, “University”, “College”, or “Convocation”).
        5. EXPERIENCE_CERTIFICATE — Employment experience/relieving letter issued by a company or employer.
        6. SERVICE_LETTER — Service certificate or employment verification letter confirming employment details or duration.
        7. APPLICATION_FORM — Job or admission form layout with multiple labeled fields for name, address, etc.
        8. PHOTO — A standalone passport-size or profile photo of a person (no text, logos, or layout).
        9. AFFIDAVIT — Legal document containing the words "Affidavit", "Notary", "Oath", or "Declaration".
        10. RESUME — A resume or CV document listing personal details, education, skills, and experience.
        11. OTHER — Any document not matching the above, including: typing certificates, short training course certificates, workshop completion certificates, computer course certificates, or any non-formal educational certificates.

        **Rules:**
        - Output must be one of the exact category names above (e.g., `AADHAR_CARD`).
        - Do NOT include explanations, reasons, or extra text.
        - If uncertain, choose `OTHER`.

        Return ONLY the category name (e.g., "AADHAR_CARD"), nothing else."""
        
        classifications = {}
        
        for i, img in enumerate(images, 1):
            try:
                if streamer:
                    streamer.send_update("classification_processing", {
                        "agent": "classification",
                        "tool": "ClassifyDocumentsTool",
                        "message": f"Checking {img.name} ({i}/{len(images)})",
                        "status": "Classifying your documents now…",
                        "progress": i / len(images) * 100
                    })
                
                logger.debug("[ClassifyTool] [%d/%d] Classifying %s", i, len(images), img.name)
                
                img_data = self._encode_image(img)
                ext = img.suffix.lower().lstrip('.')
                mime = f"image/{ext if ext in ['png', 'jpeg', 'jpg'] else 'jpeg'}"
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": classification_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{img_data}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }],
                    temperature=0.1,
                    max_tokens=50,
                )
                
                doc_type = response.choices[0].message.content.strip().upper()
                # Validate it's one of our known types
                valid_types = {
                    "AADHAR_CARD", "PAN_CARD", "MARKSHEET", "DEGREE_CERTIFICATE", "EXPERIENCE_CERTIFICATE", "SERVICE_LETTER", "APPLICATION_FORM", "PHOTO", "AFFIDAVIT", "RESUME", "OTHER"
                }
                if doc_type not in valid_types:
                    doc_type = "OTHER"
                
                classifications[img.name] = doc_type
                logger.info("Classified %s as %s", img.name, doc_type)
                
            except Exception as e:
                logger.exception("Error classifying %s: %s", img.name, e)
                classifications[img.name] = "OTHER"
                continue
        
        # Save classification results
        try:
            classification_path = data_store_dir / "document_classifications.json"
            with classification_path.open("w", encoding="utf-8") as f:
                json.dump(classifications, f, ensure_ascii=False, indent=2)
            logger.info("[ClassifyTool] Saved classifications to %s", classification_path)
            logger.info("[ClassifyTool] classifications are %s", classifications)
        except Exception as e:
            logger.exception("[ClassifyTool] Failed to save classifications: %s", e)
        
        if streamer:
            streamer.send_update("classification_complete", {
                "agent": "classification",
                "tool": "ClassifyDocumentsTool",
                "message": f"All documents have been identified.",
                "summary": f"classification summary: {classifications}",
                "status": "Classification complete!",
                "total_images": len(images),
                "classifications": classifications
            })
        logger.info("[ClassifyTool] Classification complete")

        return json.dumps(classifications)
