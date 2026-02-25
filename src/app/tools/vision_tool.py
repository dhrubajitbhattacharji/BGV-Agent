import json
import base64
import os
import logging
from pathlib import Path
from typing import List, Type, Dict, Any, Optional
from io import BytesIO

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False

logger = logging.getLogger(__name__)

def _verhoeff_check(num_str: str) -> bool:
    """Verify Aadhaar number using Verhoeff algorithm."""
    if not num_str or len(num_str) != 12:
        return False
    
    # Check for sequential or repetitive patterns
    if num_str in ["123456789012", "111111111111", "222222222222", "333333333333", "444444444444", "555555555555", "666666666666", "777777777777", "888888888888", "999999999999", "000000000000"]:
        return False
    
    # Check for simple sequential patterns
    if all(int(num_str[i]) == int(num_str[i-1]) for i in range(1, 12)):
        return False
    
    # Verhoeff algorithm tables
    mul = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,2,3,4,0,6,7,8,9,5],
        [2,3,4,0,1,7,8,9,5,6],
        [3,4,0,1,2,8,9,5,6,7],
        [4,0,1,2,3,9,5,6,7,8],
        [5,9,8,7,6,0,4,3,2,1],
        [6,5,9,8,7,1,0,4,3,2],
        [7,6,5,9,8,2,1,0,4,3],
        [8,7,6,5,9,3,2,1,0,4],
        [9,8,7,6,5,4,3,2,1,0]
    ]
    perm = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,5,7,6,2,8,3,0,9,4],
        [5,8,0,3,7,9,6,1,4,2],
        [8,9,1,6,0,4,3,5,2,7],
        [9,4,5,3,1,2,6,8,7,0],
        [4,2,8,6,5,7,3,9,0,1],
        [2,7,9,3,8,0,6,4,1,5],
        [7,0,4,6,9,1,3,2,5,8]
    ]
    
    c = 0
    for i, ch in enumerate(reversed(num_str)):
        d = ord(ch) - 48
        c = mul[c][perm[i % 8][d]]
    return c == 0

def _validate_and_format_aadhaar(value: str) -> Optional[str]:
    """Validate and format Aadhaar number."""
    if not value or not isinstance(value, str):
        return None
    
    # Remove all non-digits
    import re
    digits = re.sub(r"\D", "", value)
    
    # Check if it's 12 digits and passes Verhoeff check
    if len(digits) == 12 and _verhoeff_check(digits):
        return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"
    
    return None

class ExtractFromImagesInput(BaseModel):
    employee_id: str = Field(..., description="Employee identifier whose staged folder is data_store/{employee_id}.")

class ExtractStructuredDataFromImagesTool(BaseTool):
    name: str = "extract_structured_data_from_images"
    description: str = (
        "Send all images from data_store/{employee_id} directly to a vision-capable LLM "
        "to extract structured employee verification data. Returns JSON with all required fields."
    )
    args_schema: Type[BaseModel] = ExtractFromImagesInput

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for LLM vision API.
        Behavior:
        - If Pillow is available, resize the image so the longest side is at most IMAGE_MAX_DIM (env, default 1024) and compress JPEG/PNG to reduce token cost. Quality is controlled by IMAGE_QUALITY (env, default 75).
        - If Pillow is not available, return raw base64 (same as before).
        """
        max_dim = int(os.getenv("IMAGE_MAX_DIM", "1024"))
        quality = int(os.getenv("IMAGE_QUALITY", "75"))

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
        """Send images to LLM vision API and extract structured data using document classifications."""
        import time
        start_time = time.time()
        streamer = getattr(self, '_streamer', None)
        employee_id = employee_id.strip().rstrip('}').rstrip('"').strip()
        try:
            root = Path(__file__).resolve().parents[3]
            # Import markdown formatter
            try:
                import sys
                src_path = str(root / "src")
                if src_path not in sys.path:
                    sys.path.append(src_path)
                from app.tools.markdown_formatter import format_employee_data_as_markdown
            except ImportError:
                logger.warning("[VisionTool] Could not import markdown_formatter")
                format_employee_data_as_markdown = None
        except Exception:
            root = Path.cwd()
            format_employee_data_as_markdown = None
        staged = root / "data_store" / employee_id

        if not staged.exists() or not staged.is_dir():
            error_msg = f"Files not found for employee {employee_id}"
            if streamer:
                streamer.send_update("vision_error", {
                    "agent": "vision_extractor",
                    "tool": "ExtractStructuredDataFromImagesTool",
                    "status": "Error: Employee files not found",
                    "error": error_msg,
                    "start_timestamp": start_time,
                    "end_timestamp": time.time()
                })
            return json.dumps({
                "error": error_msg,
                "employee_id": employee_id
            })

        classifications_path = staged / "document_classifications.json"
        classifications = {}
        if classifications_path.exists():
            try:
                with classifications_path.open("r", encoding="utf-8") as f:
                    classifications = json.load(f)
                logger.info("[VisionTool] Loaded classifications for %d documents", len(classifications))
            except Exception as e:
                logger.warning("[VisionTool] Warning: Could not load classifications: %s", e)
        
        images = sorted([
            f for f in staged.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])

        if not images:
            error_msg = "No images found for processing"
            if streamer:
                streamer.send_update("vision_error", {
                    "agent": "vision_extractor",
                    "tool": "ExtractStructuredDataFromImagesTool",
                    "status": "Error: No images available",
                    "error": error_msg,
                    "start_timestamp": start_time,
                    "end_timestamp": time.time()
                })
            return json.dumps({
                "error": error_msg,
                "employee_id": employee_id
            })

        if streamer:
            streamer.send_update("vision_analysis", {
                "agent": "vision_extractor",
                "tool": "ExtractStructuredDataFromImagesTool",
                "input": f"Analyzing images for employee {employee_id}",
                "status": "Preparing for analysis",
                "message": "Your documents are ready for AI extraction.",
                "total_images": len(images),
                "start_timestamp": start_time,
                "current_timestamp": time.time()
            })

        # Import LLM client lazily so module import doesn't require it
        try:
            from openai import OpenAI
        except Exception:
            OpenAI = None
            return json.dumps({"error": "OpenAI client unavailable"})

        # OCR Client Configuration (Nanonets)
        ocr_api_key = os.getenv("OCR_API_KEY")
        ocr_base_url = os.getenv("OCR_BASE_URL")
        ocr_model = os.getenv("OCR_MODEL", "nanonets/Nanonets-OCR2-3B")

        if not ocr_api_key:
            return json.dumps({"error": "OCR_API_KEY not set"})

        ocr_client = OpenAI(api_key=ocr_api_key, base_url=ocr_base_url) if ocr_base_url else OpenAI(api_key=ocr_api_key)
        logger.info("[VisionTool] Using OCR API at %s with model %s", ocr_base_url or "default", ocr_model)

        # Summarizer Client Configuration (gpt-oss-20b)
        summarizer_api_key = os.getenv("FORMATTER_LLM_API_KEY")
        summarizer_base_url = os.getenv("FORMATTER_LLM_BASE_URL")
        summarizer_model_name = os.getenv("FORMATTER_LLM_MODEL", "mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit")

        summarizer_client = None
        if summarizer_api_key:
            summarizer_client = OpenAI(api_key=summarizer_api_key, base_url=summarizer_base_url) if summarizer_base_url else OpenAI(api_key=summarizer_api_key)
            logger.info("[VisionTool] Using Summarizer API at %s with model %s", summarizer_base_url or "default", summarizer_model_name)
        else:
            logger.warning("[VisionTool] SUMMARIZER_LLM_API_KEY not set, summarizer will be unavailable")

        # JSON Generator Client Configuration (Qwen2.5-VL-7B-Instruct)
        # json_gen_api_key = os.getenv("LLM_API_KEY")
        # json_gen_base_url = os.getenv("LLM_BASE_URL")
        # json_gen_model_name = os.getenv("VISION_MODEL", "mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit")
        
        json_gen_api_key = os.getenv("FORMATTER_LLM_API_KEY")
        json_gen_base_url = os.getenv("FORMATTER_LLM_BASE_URL")
        json_gen_model_name = os.getenv("FORMATTER_LLM_MODEL", "mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit")

        json_gen_client = None
        if json_gen_api_key:
            json_gen_client = OpenAI(api_key=json_gen_api_key, base_url=json_gen_base_url) if json_gen_base_url else OpenAI(api_key=json_gen_api_key)
            logger.info("[VisionTool] Using JSON Generator API at %s with model %s", json_gen_base_url or "default", json_gen_model_name)
        else:
            logger.warning("[VisionTool] JSON_GEN_LLM_API_KEY not set, JSON generator will be unavailable")

        prompts_by_type = {
            "AADHAR_CARD": """Extract ALL visible text and data fields from this Aadhaar card image.
            Return a flat JSON object where keys are the field labels (normalized to snake_case) and values are the extracted text.
            Include fields like: name, aadhar_no, fathers_name, address, date_of_birth, gender, etc.
            Do not invent fields. Only extract what is visible.""",

            "PAN_CARD": """Extract ALL visible text and data fields from this PAN card image.
            Return a flat JSON object where keys are the field labels (normalized to snake_case) and values are the extracted text.
            Include fields like: name, pan_no, fathers_name, date_of_birth, etc.
            Do not invent fields. Only extract what is visible.""",

            "MARKSHEET": """Extract ALL visible text and data fields from this marksheet/transcript.
            Return a JSON object with keys for student details, institution details, and a list or object for marks/grades.
            Include fields like: name, roll_no, course_name, college_name, session/year, subjects (as a list or dict), total_marks, percentage, result, etc.
            Do not invent fields. Only extract what is visible.""",

            "DEGREE_CERTIFICATE": """Extract ALL visible text and data fields from this degree certificate.
            Return a JSON object with keys for student details, degree details, and institution details.
            Include fields like: name, course_name, college_name, university, date_of_issue, division/class, etc.
            Do not invent fields. Only extract what is visible.""",

            "EXPERIENCE_CERTIFICATE": """Extract ALL visible text and data fields from this experience certificate.
            Return a JSON object with keys for employee details, company details, and employment period.
            Include fields like: name, designation, company_name, joining_date, relieving_date, employment_period, etc.
            Do not invent fields. Only extract what is visible.""",

            "SERVICE_LETTER": """Extract ALL visible text and data fields from this service letter.
            Return a JSON object with keys for employee details and service details.
            Include fields like: name, employee_code, designation, department, joining_date, etc.
            Do not invent fields. Only extract what is visible.""",

            "APPLICATION_FORM": """Extract ALL visible text and data fields from this application form.
            Return a JSON object capturing all filled information.
            Include fields like: name, personal_details, contact_info, educational_background, work_experience, etc.
            Do not invent fields. Only extract what is visible.""",
            
            "RESUME": """Extract ALL visible text and data fields from this resume/CV.
            Return a JSON object with structured sections.
            Include fields like: personal_info, summary, skills, education (list), experience (list), projects, certifications, etc.
            Do not invent fields. Only extract what is visible.""",

            "AFFIDAVIT": """Extract ALL visible text and data fields from this affidavit.
            Return a JSON object capturing the deponent's details and the sworn statements.
            Include fields like: deponent_name, father_name, address, purpose, date, verification, etc.
            Do not invent fields. Only extract what is visible.""",
        }

        generic_prompt = """Extract ALL visible text and data fields from this document. Return a valid JSON object representing the content.
        Use snake_case for keys. Capture every piece of meaningful information visible in the image. Do not invent fields. Only extract what is visible."""

        all_extractions: List[Dict[str, Any]] = []
        import re

        max_images = int(os.getenv("VISION_MAX_IMAGES", "150"))  # Increased default from 8 to 50
        logger.info("Processing %d images for employee %s", len(images[:max_images]), employee_id)

        if streamer:
            streamer.send_update("vision_processing", {
                "agent": "vision_extractor",
                "tool": "ExtractStructuredDataFromImagesTool",
                "input": f"Processing {min(len(images), max_images)} images for employee {employee_id}",
                "status": "Starting image analysis with vision AI",
                "message": "Starting document-aware image analysis",
                "total_to_process": min(len(images), max_images),
                "start_timestamp": start_time,
                "current_timestamp": time.time()
            })

        # STEP 1: Extract OCR text from all documents first
        ocr_results: List[Dict[str, Any]] = []
        
        for i, img in enumerate(images[:max_images], 1):
            try:
                if streamer:
                    streamer.send_update("vision_ocr_processing", {
                        "agent": "vision_extractor",
                        "tool": "ExtractStructuredDataFromImagesTool",
                        "input": f"OCR extraction {i} of {min(len(images), max_images)}",
                        "status": "Extracting text from documents…",
                        "message": f"Parsing document {i}/{min(len(images), max_images)}",
                        "progress": i / min(len(images), max_images) * 100,
                        "start_timestamp": start_time,
                        "current_timestamp": time.time()
                    })

                # Get document type
                doc_type = classifications.get(img.name, "OTHER")
                logger.debug("OCR processing image %d/%d: %s (%s)", i, min(len(images), max_images), img.name, doc_type)

                img_data = self._encode_image(img)
                ext = img.suffix.lower().lstrip('.')
                mime = f"image/{ext if ext in ['png', 'jpeg', 'jpg'] else 'jpeg'}"
                
                # OCR extraction using Nanonets
                ocr_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
                
                ocr_vision_content = [
                    {"type": "text", "text": ocr_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{img_data}",
                            "detail": "low"
                        }
                    }
                ]

                try:
                    logger.debug("[VisionTool] Step 1: Extracting OCR with %s", ocr_model)
                    ocr_response = ocr_client.chat.completions.create(
                        model=ocr_model,
                        messages=[{"role": "user", "content": ocr_vision_content}],
                        temperature=0.1,
                        max_tokens=4096,
                    )
                    raw_ocr_text = ocr_response.choices[0].message.content or ""
                    logger.info("[VisionTool] Nanonets OCR extracted %d chars from %s", len(raw_ocr_text), img.name)
                    
                    if not raw_ocr_text:
                        logger.warning("[VisionTool] No OCR text extracted from %s", img.name)
                        ocr_results.append({
                            "file": img.name,
                            "doc_type": doc_type,
                            "error": "No text extracted by OCR"
                        })
                        continue
                    
                    # Clean OCR text - remove watermarks and page numbers
                    cleaned_ocr = raw_ocr_text
                    cleaned_ocr = re.sub(r'<watermark>.*?</watermark>\s*', '', cleaned_ocr, flags=re.DOTALL)
                    cleaned_ocr = re.sub(r'<page_number>.*?</page_number>\s*', '', cleaned_ocr, flags=re.DOTALL)
                    
                    ocr_results.append({
                        "file": img.name,
                        "doc_type": doc_type,
                        "ocr_text": cleaned_ocr
                    })

                except Exception as e:
                    logger.exception("[VisionTool] Failed OCR extraction for %s: %s", img.name, e)
                    ocr_results.append({
                        "file": img.name,
                        "doc_type": doc_type,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    continue

            except Exception as outer_e:
                logger.exception("[VisionTool] Unexpected error during OCR for %s: %s", img.name, outer_e)
                ocr_results.append({
                    "file": img.name,
                    "error": str(outer_e),
                    "error_type": type(outer_e).__name__
                })
                continue

        # Check if we have any successful OCR extractions
        successful_ocr = [r for r in ocr_results if "ocr_text" in r]
        if not successful_ocr:
            error_msg = "No OCR text could be extracted from any images"
            logger.error("[VisionTool] %s", error_msg)
            if streamer:
                streamer.send_update("vision_error", {
                    "agent": "vision_extractor",
                    "tool": "ExtractStructuredDataFromImagesTool",
                    "status": "Error: OCR extraction failed for all images",
                    "error": error_msg,
                    "start_timestamp": start_time,
                    "end_timestamp": time.time()
                })
            return json.dumps({"error": error_msg, "employee_id": employee_id})

        logger.info("[VisionTool] Completed OCR extraction for %d/%d documents", len(successful_ocr), len(ocr_results))
        
        # STEP 2: Clean and prepare OCR text from all documents
        if streamer:
            streamer.send_update("vision_json_generation", {
                "agent": "vision_extractor",
                "tool": "ExtractStructuredDataFromImagesTool",
                "input": f"Generating consolidated JSON from {len(successful_ocr)} documents",
                "status": "Creating structured JSON…",
                "message": "Extracting structured data directly from OCR text",
                "start_timestamp": start_time,
                "current_timestamp": time.time()
            })
        
        # First, create cleaned document texts
        document_texts = []
        for i, ocr_result in enumerate(successful_ocr, 1):
            doc_type = ocr_result["doc_type"]
            ocr_text = ocr_result["ocr_text"]
            file_name = ocr_result["file"]
            
            # Clean the OCR text
            import re
            clean_text = re.sub(r'<[^>]+>', '', ocr_text)  # Remove HTML tags
            clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean_text) # Remove non-printable chars
            clean_text = re.sub(r'\*{2,}', '', clean_text)  # Remove markdown bold
            clean_text = re.sub(r'\s+', ' ', clean_text) # Collapse whitespace
            clean_text = clean_text.strip()
            
            if len(clean_text) > 50:
                document_texts.append(f"Document Type: {doc_type}\nContent: {clean_text}\n")
        
        # Combine all document texts
        combined_ocr_text = "\n\n".join(document_texts)
        logger.info("[VisionTool] Combined OCR text total: %d chars", len(combined_ocr_text))
        
        # STEP 3: Generate consolidated JSON directly from OCR text
        consolidated_json = {}
        if json_gen_client and combined_ocr_text:
            logger.debug("[VisionTool] Step 2: Generating consolidated JSON directly from OCR with %s", json_gen_model_name)
            
            json_prompt = f"""You are an Employee Background Verification Agent. Extract and consolidate information from these employee verification documents into a well-structured FLAT JSON object.

            IMPORTANT RULES:
            1. Create a FLAT structure where all fields are at the root level. DO NOT group fields by document type.
            2. Personal information fields directly at root: name, father_name, mother_name, date_of_birth, gender, etc.
            3. ID numbers directly at root: aadhaar_number, pan_number, passport_number, etc.
            4. Contact info directly at root: mobile_number, email, address (as nested object), etc.
            5. Use arrays for repeating items: education_details (array of objects), employment_details (array of objects)
            6. Use snake_case for all keys
            7. Merge duplicate information intelligently (if same info appears in multiple docs, use the most complete version)
            8. Only include information that is actually present in the documents
            9. Return ONLY valid JSON, no explanations

            Required JSON Structure:
            {{
                "name": "string or null",
                "father_name": "string or null",
                "mother_name": "string or null",
                "date_of_birth": "YYYY-MM-DD format or null",
                "gender": "string or null",
                "aadhaar_number": "#### #### #### format or null",
                "pan_number": "string or null",
                "mobile_number": "string or null",
                "email": "string or null",
                "address": "string or null",
                "education_details": [
                    {{
                        "course_name": "string",
                        "institution_name": "string",
                        "course_tenure": "string",
                        "college_address": "string or null",
                        "percentage_obtained": "string or null"
                    }}
                ],
                "employment_details": [
                    {{
                        "employment_start_date": "YYYY-MM-DD or null",
                        "employment_end_date": "YYYY-MM-DD or null",
                        "company_name": "string",
                        "company_address": "string or null",
                        "date_of_joining": "YYYY-MM-DD or null"
                    }}
                ]
            }}

            DOCUMENTS:
            {combined_ocr_text}

            Return only valid FLAT JSON (no document type grouping):"""
            
            try:
                json_resp = json_gen_client.chat.completions.create(
                    model=json_gen_model_name,
                    messages=[{"role": "user", "content": json_prompt}],
                    temperature=0.1,
                    max_tokens=4000,
                )
                json_result = json_resp.choices[0].message.content or ""
                logger.info("[VisionTool] Generated consolidated JSON response (%d chars)", len(json_result))
                
                # Try to parse JSON from response with multiple fallback strategies
                try:
                    # Extract JSON from markdown code blocks if present
                    if json_result and "```" in json_result:
                        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", json_result)
                        if match:
                            json_result = match.group(1).strip()
                    
                    # Try direct parsing
                    try:
                        consolidated_json = json.loads(json_result) if json_result else {}
                        logger.info("[VisionTool] Successfully parsed consolidated JSON with %d top-level keys", len(consolidated_json))
                    except json.JSONDecodeError as e:
                        logger.warning("[VisionTool] First JSON parse failed: %s", e)
                        logger.debug("[VisionTool] Raw JSON result: %s", json_result[:1000])
                        
                        # Fallback 1: Try to fix common JSON issues
                        try:
                            fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_result)
                            fixed_json = fixed_json.replace("'", '"')
                            consolidated_json = json.loads(fixed_json)
                            logger.info("[VisionTool] Successfully parsed JSON after fixing common issues")
                        except json.JSONDecodeError as e2:
                            logger.warning("[VisionTool] JSON fix attempt failed: %s", e2)
                            
                            # Fallback 2: Try to extract any valid JSON object
                            try:
                                json_match = re.search(r'\{[\s\S]*\}', json_result)
                                if json_match:
                                    potential_json = json_match.group(0)
                                    potential_json = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                                    consolidated_json = json.loads(potential_json)
                                    logger.info("[VisionTool] Successfully extracted and parsed JSON")
                                else:
                                    raise json.JSONDecodeError("No JSON object found", json_result, 0)
                            except (json.JSONDecodeError, AttributeError) as e3:
                                logger.error("[VisionTool] All JSON parse attempts failed: %s", e3)
                                consolidated_json = {
                                    "parse_error": "Failed to parse JSON",
                                    "raw_response": json_result[:500]
                                }
                except Exception as parse_error:
                    logger.error("[VisionTool] Unexpected error during JSON parsing: %s", parse_error)
                    consolidated_json = {"error": "JSON parsing exception", "details": str(parse_error)}
            except Exception as e:
                logger.error("[VisionTool] Failed to generate consolidated JSON: %s", e)
                consolidated_json = {"error": str(e)}
        else:
            logger.warning("[VisionTool] No JSON generator client available")
            consolidated_json = {"error": "JSON generator not configured"}
        
        # Store per-document OCR results for debugging/reference
        for ocr_result in successful_ocr:
            all_extractions.append(ocr_result)
        
        logger.info("[VisionTool] Completed consolidated JSON generation")

        if not all_extractions:
            error_msg = "No images could be processed"
            if streamer:
                streamer.send_update("vision_error", {
                    "agent": "vision_extractor",
                    "tool": "ExtractStructuredDataFromImagesTool",
                    "status": "Error: No images processed successfully",
                    "error": error_msg,
                    "start_timestamp": start_time,
                    "end_timestamp": time.time()
                })
            return json.dumps({"error": error_msg, "employee_id": employee_id})

        # Use the consolidated JSON
        data = consolidated_json
        
        # STEP 4: Generate markdown from the consolidated JSON
        if streamer:
            streamer.send_update("vision_markdown_generation", {
                "agent": "vision_extractor",
                "tool": "ExtractStructuredDataFromImagesTool",
                "input": "Generating markdown report from consolidated JSON",
                "status": "Creating markdown report…",
                "message": "Formatting structured data as markdown",
                "start_timestamp": start_time,
                "current_timestamp": time.time()
            })
        
        markdown_content = ""
        if format_employee_data_as_markdown and isinstance(data, dict) and "error" not in data:
            try:
                markdown_content = format_employee_data_as_markdown(data)
                logger.info("[VisionTool] Generated markdown from consolidated JSON (%d chars)", len(markdown_content))
            except Exception as e:
                logger.warning("[VisionTool] Failed to generate markdown from JSON: %s", e)
                markdown_content = f"# Employee Verification Report\n\nStructured data extraction completed.\n\nSee structured.json for details."
        else:
            logger.warning("[VisionTool] Markdown formatter not available or invalid data")
            markdown_content = f"# Employee Verification Report\n\nStructured data extraction completed.\n\nSee structured.json for details."

        try:
            out_dir = root / "data_store" / employee_id
            out_dir.mkdir(parents=True, exist_ok=True)

            structured_path = out_dir / "structured.json"
            logger.info("[VisionTool] Saving structured.json to %s", structured_path)
            with structured_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("[VisionTool] Saved structured.json (%d bytes)", structured_path.stat().st_size)

            # Save markdown version
            try:
                markdown_path = out_dir / "structured.md"
                with markdown_path.open("w", encoding="utf-8") as f:
                    f.write(markdown_content)
                logger.info("[VisionTool] Saved structured.md (%d bytes)", markdown_path.stat().st_size)
            except Exception as md_error:
                logger.warning("[VisionTool] Failed to save markdown: %s", md_error)

            vision_extractions_path = out_dir / "vision_extractions.json"
            logger.info("[VisionTool] Saving vision_extractions.json to %s", vision_extractions_path)
            with vision_extractions_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "consolidated_data": data,
                    "consolidated_markdown": markdown_content,
                    "per_image_ocr": all_extractions,
                    "total_images": len(images),
                    "processed_images": len(all_extractions),
                }, f, ensure_ascii=False, indent=2)
            logger.info("[VisionTool] Saved vision_extractions.json (%d bytes)", vision_extractions_path.stat().st_size)

        except Exception as e:
            logger.exception("[VisionTool] Failed to save JSON files: %s", e)

        # Return the consolidated data directly without employee_id or document names
        result = {
            "success": True,
            "data": data,
            "markdown_report": markdown_content
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
