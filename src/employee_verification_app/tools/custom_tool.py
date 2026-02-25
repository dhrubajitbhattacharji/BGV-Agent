import json
import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Type, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

class IngestEmployeeFilesInput(BaseModel):
    source_folder: str = Field(..., description="Path to the incoming folder containing uploaded PDFs and images (e.g., /path/to/incoming/abc123).")
    employee_id: Optional[str] = Field(None, description="Employee identifier - optional here; will be resolved from metadata or source_folder when missing.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata container where employee_id may be provided.")

class IngestEmployeeFilesTool(BaseTool):
    name: str = "ingest_employee_files"
    description: str = (
        "Process employee documents from the incoming folder and save to data_store. "
        "REQUIRED PARAMETERS: source_folder (incoming folder path) AND employee_id (employee request ID). "
        "Converts PDF pages to PNG images and copies image files to data_store/{employee_id}/. "
        "Returns JSON with processing results including count of images created."
    )
    args_schema: Type[BaseModel] = IngestEmployeeFilesInput

    def _run(self, source_folder: str, employee_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        streamer = getattr(self, '_streamer', None)

        if not employee_id and isinstance(metadata, dict):
            for key in ("employee_id", "request_id", "requestId", "id"):
                if key in metadata and metadata[key]:
                    employee_id = str(metadata[key])
                    break

        if not employee_id:
            try:
                inferred = Path(source_folder).name
                if inferred:
                    employee_id = inferred
            except Exception:
                employee_id = None

        if not employee_id:
            error_msg = "employee_id not provided and could not be inferred from metadata or source_folder"
            logger.error("[IngestTool] ERROR: %s", error_msg)
            return json.dumps({"error": error_msg})

        incoming_dir = Path(source_folder).expanduser().resolve()
        logger.info("[IngestTool] Starting file processing for employee: %s", employee_id)
        logger.debug("[IngestTool] Incoming directory: %s", incoming_dir)

        if not incoming_dir.exists() or not incoming_dir.is_dir():
            error_msg = f"Incoming folder not found: {incoming_dir}"
            logger.error("[IngestTool] ERROR: %s", error_msg)
            return json.dumps({"error": error_msg})

        try:
            root = Path(__file__).resolve().parents[3]
        except Exception:
            root = Path.cwd()
        data_store_dir = root / "data_store" / employee_id
        data_store_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("[IngestTool] Data store directory: %s", data_store_dir)

        all_files = [f for f in incoming_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
        logger.info("[IngestTool] Found %d files to process", len(all_files))

        if streamer:
            streamer.send_update("ingest_start", {
                "agent": "ingest_files",
                "tool": "IngestEmployeeFilesTool",
                "message": f"We found {len(all_files)} files. Getting things ready…",
                "status": "Starting file ingestion…",
                "total_files": len(all_files)
            })

        processed_images: List[str] = []
        pdf_images_created = 0
        
        for file_idx, entry in enumerate(all_files, 1):
            ext = entry.suffix.lower()
            
            if streamer:
                streamer.send_update("ingest_processing", {
                    "agent": "ingest_files",
                    "tool": "IngestEmployeeFilesTool",
                    "message": f"Reading {entry.name} ({file_idx}/{len(all_files)})",
                    "status": "Processing your files now…",
                    "progress": file_idx / len(all_files) * 100
                })
            
            if ext == ".pdf":
                # Convert PDF pages to PNG images in data_store
                try:
                    import pypdfium2 as pdfium
                    pdf = pdfium.PdfDocument(str(entry))
                    n_pages = len(pdf)
                    base = entry.stem
                    
                    logger.info("[IngestTool] Converting PDF %s (%d pages)...", entry.name, n_pages)
                    
                    for i in range(n_pages):
                        page = pdf[i]
                        bmp = page.render(scale=2.0)
                        pil = bmp.to_pil()
                        
                        image_name = f"{base}_page_{i+1:03d}.png"
                        image_path = data_store_dir / image_name
                        pil.save(image_path, format="PNG")
                        
                        processed_images.append(str(image_path))
                        pdf_images_created += 1
                    
                    logger.info("[IngestTool] Created %d images from PDF in data_store", n_pages)
                    
                except Exception as e:
                    logger.exception("[IngestTool] Failed to process PDF %s: %s", entry.name, e)
                    continue
            else:
                # Copy image files to data_store
                dest = data_store_dir / entry.name
                try:
                    shutil.copy2(entry, dest)
                    processed_images.append(str(dest))
                    logger.info("[IngestTool] Copied %s to data_store", entry.name)
                except Exception as e:
                    logger.exception("[IngestTool] Failed to copy %s: %s", entry.name, e)
                    continue

        result = {
            "data_store_folder": str(data_store_dir),
            "processed_images": len(processed_images),
            "pdf_images_created": pdf_images_created,
            "total_files": len(processed_images)
        }

        logger.info("[IngestTool] Processing complete: %d files in data_store", len(processed_images))
        if streamer:
            streamer.send_update("ingest_complete", {
                "agent": "ingest_files",
                "tool": "IngestEmployeeFilesTool",
                "message": f"All files processed successfully — {len(processed_images)} images prepared for analysis.",
                "status": "Ingestion complete!",
                "total_images": len(processed_images),
                "pdf_images_created": pdf_images_created
            })

        return json.dumps(result)

class ExtractTextForEmployeeInput(BaseModel):
    employee_id: str = Field(..., description="Employee identifier whose staged folder is data_store/{employee_id}.")

class ExtractTextForEmployeeTool(BaseTool):
    name: str = "extract_text_for_employee"
    description: str = (
        "Extract text from all PDFs and images under data_store/{employee_id} using vision LLM. "
        "Returns JSON array of {filename, text}."
    )
    args_schema: Type[BaseModel] = ExtractTextForEmployeeInput

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF using PyPDF library."""
        try:
            from pypdf import PdfReader
        except Exception as e:
            return f"[PDF extraction unavailable: {e}]"

        try:
            reader = PdfReader(str(file_path))
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    pass
            texts: List[str] = []
            for page in reader.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    texts.append("")
            return "\n".join(texts).strip()
        except Exception as e:
            return f"[PDF read error: {e}]"

    def _extract_image_text_with_llm(self, file_path: Path) -> str:
        """Extract text from image using vision LLM."""
        try:
            import os
            import base64
            from openai import OpenAI

            with file_path.open("rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            ext = file_path.suffix.lower().lstrip('.')
            mime = f"image/{ext if ext in ['png', 'jpeg', 'jpg'] else 'jpeg'}"
            
            api_key = os.getenv("LLM_API_KEY")
            base_url = os.getenv("LLM_BASE_URL")
            model = os.getenv("VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
            
            if not api_key:
                return "[LLM_API_KEY not set]"
            
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract ALL text from this image. Return ONLY the extracted text with no additional commentary or formatting. Preserve the original layout and line breaks as much as possible."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }],
                temperature=0.1,
                max_tokens=2000,
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            return f"[Vision LLM extraction error: {e}]"

    def _run(self, employee_id: str) -> str:
        """Extract text from all files in employee's data_store folder."""
        try:
            root = Path(__file__).resolve().parents[3]
        except Exception:
            root = Path.cwd()
        
        staged = root / "data_store" / employee_id
        if not staged.exists() or not staged.is_dir():
            return json.dumps({
                "error": f"Staged folder not found for employee_id {employee_id}",
                "staged_folder": str(staged)
            })

        results = []
        for f in sorted(staged.iterdir()):
            if not f.is_file():
                continue
            
            ext = f.suffix.lower()
            text = ""
            
            if ext == ".pdf":
                text = self._extract_pdf_text(f)
            elif ext in {".jpg", ".jpeg", ".png"}:
                text = self._extract_image_text_with_llm(f)
            else:
                continue
            
            results.append({"filename": f.name, "text": text})

        return json.dumps(results)
