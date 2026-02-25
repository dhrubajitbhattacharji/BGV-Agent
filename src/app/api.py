import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from langfuse import Langfuse, observe, get_client
from openinference.instrumentation.crewai import CrewAIInstrumentor
from dotenv import load_dotenv
from .crew import build_crew

load_dotenv()

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

langfuse = get_client()

CrewAIInstrumentor().instrument(skip_dep_check=True)

app = FastAPI(title="Employee Verification API")

def get_project_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current.parent

@observe()
def _process_and_push(request_id: str, incoming_dir_str: str):
    """Background task: process files using CrewAI flow with vision extraction and publish results."""
    incoming_dir = Path(incoming_dir_str)
    
    print(f"[API] Processing started for request_id: {request_id}")
    print(f"[API] Incoming directory: {incoming_dir}")
    
    from app.streaming import StreamingEmployeeVerificationApp, FayeStreamer
    import json as _json

    streamer = FayeStreamer(request_id)
    root = get_project_root()
    
    # Retry logic with max 3 attempts using CrewAI flow
    max_retries = 3
    data = None
    trace = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[API] Attempt {attempt}/{max_retries} - Starting CrewAI processing...")
            
            # Initialize streaming crew wrapper
            streaming_app = StreamingEmployeeVerificationApp(request_id)
            
            # Prepare inputs for the crew
            inputs = {
                "source_folder": str(incoming_dir),
                "employee_id": request_id
            }
            
            # Execute the crew with streaming
            print(f"[API] Kicking off crew with inputs: {inputs}")
            crew_result = streaming_app.kickoff_with_streaming(inputs)
            
            print(f"[API] Crew execution completed. Result type: {type(crew_result)}")
            print(f"[API] Crew result preview: {str(crew_result)[:200] if crew_result else 'None'}")
            
            # Check if structured.json was created
            structured_path = root / "data_store" / request_id / "structured.json"
            vision_path = root / "data_store" / request_id / "vision_extractions.json"
            
            if structured_path.exists():
                with structured_path.open("r", encoding="utf-8") as f:
                    data = _json.load(f)
                
                fields_extracted = len([k for k, v in data.items() if v is not None and str(v).strip()])
                
                if fields_extracted > 0:
                    print(f"[API] ✓ Success! Extracted {fields_extracted} fields on attempt {attempt}")
                    
                    trace = {
                        "employee_id": request_id, 
                        "extraction_mode": "crewai_flow", 
                        "attempts": attempt,
                        "crew_result": str(crew_result)[:500] if crew_result else None
                    }
                    
                    if vision_path.exists():
                        with vision_path.open("r", encoding="utf-8") as f:
                            vision_data = _json.load(f)
                            trace["vision_summary"] = {
                                "total_images": vision_data.get("total_images", 0),
                                "processed_images": vision_data.get("processed_images", 0),
                                "fields_extracted": fields_extracted
                            }
                    
                    # Final success notification is already sent by StreamingEmployeeVerificationApp
                    break
                else:
                    print(f"[API] Attempt {attempt} - No data extracted, retrying...")
                    if attempt < max_retries:
                        continue
                    else:
                        data = {}
                        trace = {
                            "employee_id": request_id, 
                            "extraction_mode": "crewai_flow", 
                            "error": "All fields null after max retries", 
                            "attempts": attempt
                        }
            else:
                print(f"[API] Attempt {attempt} - structured.json not found, retrying...")
                if attempt < max_retries:
                    continue
                else:
                    data = {}
                    trace = {
                        "employee_id": request_id, 
                        "extraction_mode": "crewai_flow", 
                        "error": "No output file created after max retries", 
                        "attempts": attempt
                    }
                    
        except Exception as e:
            print(f"[API] Attempt {attempt} failed: {e}")
            import traceback
            traceback.print_exc()
            
            if attempt < max_retries:
                print(f"[API] Retrying...")
                continue
            else:
                print(f"[API] All {max_retries} attempts failed")
                data = {}
                trace = {
                    "employee_id": request_id, 
                    "extraction_mode": "crewai_flow", 
                    "error": str(e)[:500], 
                    "attempts": attempt
                }
                
                streamer.send_update(
                    "processing_failed",
                    {
                        "step": attempt,
                        "agent": "crew_manager",
                        "input": f"Processing failed for employee {request_id}",
                        "status": "Processing failed after max retries",
                        "error": str(e)[:200]
                    },
                    is_final=True
                )
    
    # Cleanup incoming directory after processing
    try:
        if incoming_dir.exists():
            print(f"[API] Cleaning up incoming directory: {incoming_dir}")
            shutil.rmtree(incoming_dir, ignore_errors=True)
            print(f"[API] ✓ Incoming directory cleaned up")
    except Exception as e:
        print(f"[API] Warning: Failed to cleanup incoming directory: {e}")
    
    # Save trace
    try:
        out_dir = root / "data_store" / request_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "trace.json").open("w", encoding="utf-8") as f:
            _json.dump(trace, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Push notifications
    payload = {"request_id": request_id, "event": "result", "data": data, "trace": trace}
    
    push_url = os.getenv("PUSH_WEBHOOK_URL")
    if push_url:
        try:
            with httpx.Client(timeout=10) as client:
                client.post(push_url, json=payload)
        except Exception:
            pass

    faye_url = os.getenv("FAYE_URL", "https://faye.kriyam.ai/faye")
    faye_prefix = os.getenv("FAYE_CHANNEL_PREFIX", "").strip()
    if faye_prefix and not faye_prefix.startswith("/"):
        faye_prefix = "/" + faye_prefix
    faye_channel = f"{faye_prefix.rstrip('/')}/{request_id}" if faye_prefix else f"/{request_id}"
    faye_auth_token = os.getenv("FAYE_AUTH_TOKEN")
    if faye_url:
        try:
            message = {"channel": faye_channel, "data": payload}
            if faye_auth_token:
                message["ext"] = {"authToken": faye_auth_token}
            with httpx.Client(timeout=10) as client:
                client.post(faye_url, json=[message])
        except Exception:
            pass
    
    print(f"[API] Processing completed for {request_id}")


@app.post("/verify")
async def verify(
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = File(default=None),
    file: Optional[List[UploadFile]] = File(default=None, alias="file"),
    files_bracket: Optional[List[UploadFile]] = File(default=None, alias="files[]"),
):
    request_id = uuid.uuid4().hex
    root = get_project_root()
    incoming_dir = root / "incoming" / request_id
    incoming_dir.mkdir(parents=True, exist_ok=True)
    
    uploads: List[UploadFile] = []
    for group in (files, file, files_bracket):
        if group:
            uploads.extend(group)

    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="No files provided. Send multipart/form-data with one of: 'files', 'file', or 'files[]' keys."
        )

    saved_files = []
    for uf in uploads:
        ext = (Path(uf.filename).suffix or "").lower()
        if ext not in {".png", ".jpg", ".jpeg", ".pdf"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Only .png, .jpg, .jpeg, .pdf allowed.")
        
        dest = incoming_dir / Path(uf.filename).name
        try:
            with dest.open("wb") as f:
                content = await uf.read()
                f.write(content)
            saved_files.append(str(dest))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save {uf.filename}: {e}")

    print(f"[API] Saved {len(saved_files)} files to incoming/{request_id}")
    background_tasks.add_task(_process_and_push, request_id, str(incoming_dir))

    return JSONResponse(content={"request_id": request_id, "files_saved": len(saved_files)})


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    """Fetch the structured output for a given request_id. Returns 202 if still pending."""
    root = get_project_root()
    out_dir = root / "data_store" / request_id
    structured_path = out_dir / "structured.json"
    markdown_path = out_dir / "structured.md"
    
    if not structured_path.exists():
        return JSONResponse(status_code=202, content={"request_id": request_id, "status": "pending"})
    
    try:
        import json as _json
        data = _json.loads(structured_path.read_text(encoding="utf-8"))
        
        # Load markdown if available
        markdown_content = None
        if markdown_path.exists():
            markdown_content = markdown_path.read_text(encoding="utf-8")
        
        # Count fields (handle nested structures)
        basic_fields_count = len([k for k in ["name", "address", "date_of_birth", "fathers_name", "aadhar_no", "pan_no", "email", "phone_no"] 
                                   if data.get(k) is not None and str(data.get(k)).strip()])
        education_count = len(data.get("education_details", []))
        employment_count = len(data.get("employment_details", []))
        
        return JSONResponse(content={
            "request_id": request_id, 
            "status": "ready", 
            "basic_fields_extracted": basic_fields_count,
            "education_records": education_count,
            "employment_records": employment_count,
            "data": data,
            "markdown": markdown_content
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {e}")
    
def get_app():
    return app
