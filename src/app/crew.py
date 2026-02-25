import os
from typing import List, ClassVar, Optional

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from app.tools.custom_tool import IngestEmployeeFilesTool
from app.tools.classification_tool import ClassifyDocumentsTool
from app.tools.vision_tool import ExtractStructuredDataFromImagesTool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json
from typing import Dict, Union, Any
from dotenv import load_dotenv

load_dotenv()

temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
timeout = int(os.getenv("LLM_TIMEOUT", "180"))
max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
max_rpm = int(os.getenv("LLM_MAX_RPM", "25"))  # Rate limit to prevent context overload

local_llm = LLM(
    model=os.getenv("LLM_MODEL", "openai/Qwen/Qwen2.5-VL-7B-Instruct"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=temperature,
    timeout=timeout,
    max_retries=max_retries,
    max_rpm=max_rpm,
)

class EducationDetail(BaseModel):
    """Education qualification details."""
    course_name: Optional[str] = Field(None, description="Course or degree name (e.g., Matriculation, Graduation, B.Tech)")
    institution_name: Optional[str] = Field(None, description="Name of educational institution")
    course_tenure: Optional[str] = Field(None, description="Academic years (e.g., 2018-2022)")
    college_address: Optional[str] = Field(None, description="Institution address")
    percentage_obtained: Optional[str] = Field(None, description="Percentage or grade obtained")

class EmploymentDetail(BaseModel):
    """Employment history details."""
    employment_start_date: Optional[str] = Field(None, description="Employment start date in YYYY-MM-DD format")
    employment_end_date: Optional[str] = Field(None, description="Employment end date in YYYY-MM-DD format")
    company_name: Optional[str] = Field(None, description="Employer name")
    company_address: Optional[str] = Field(None, description="Employer address")
    date_of_joining: Optional[str] = Field(None, description="Company joining date in YYYY-MM-DD format")

class EmployeeData(BaseModel):
    """Structured employee verification data model."""
    name: Optional[str] = Field(None, description="Full name of the employee")
    address: Optional[str] = Field(None, description="Complete residential address")
    date_of_birth: Optional[str] = Field(None, description="Date of birth in YYYY-MM-DD format")
    fathers_name: Optional[str] = Field(None, description="Father's full name")
    aadhar_no: Optional[str] = Field(None, description="Aadhaar number in #### #### #### format")
    pan_no: Optional[str] = Field(None, description="PAN number")
    email: Optional[str] = Field(None, description="Email address")
    phone_no: Optional[str] = Field(None, description="Phone number")
    education_details: List[EducationDetail] = Field(default_factory=list, description="List of education qualifications")
    employment_details: List[EmploymentDetail] = Field(default_factory=list, description="List of employment history")


@CrewBase
class EmployeeVerificationApp:
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def ingestor(self) -> Agent:
        return Agent(
            config=self.agents_config["ingestor"],
            verbose=True,
            tools=[IngestEmployeeFilesTool()],
            llm=local_llm,
            allow_delegation=False,
            memory=False,
            max_iter=1,
        )

    @agent
    def classifier(self) -> Agent:
        return Agent(
            config=self.agents_config["classifier"],
            verbose=True,
            tools=[ClassifyDocumentsTool()],
            llm=local_llm,
            allow_delegation=False,
            memory=False,
            max_iter=1,
        )

    @agent
    def vision_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["vision_extractor"],
            verbose=True,
            tools=[ExtractStructuredDataFromImagesTool(), ValidateStructuredJSONTool()],
            llm=local_llm,
            allow_delegation=False,
            memory=False,
            max_iter=1,
        )

    @task
    def ingest_task(self) -> Task:
        return Task(
            config=self.tasks_config["ingest_task"],
            agent=self.ingestor(),
        )

    @task
    def classify_documents_task(self) -> Task:
        return Task(
            config=self.tasks_config["classify_documents_task"],
            agent=self.classifier(),
            context=[self.ingest_task()],
        )

    @task
    def extract_data_task(self) -> Task:
        return Task(
            config=self.tasks_config["extract_data_task"],
            agent=self.vision_extractor(),
            context=[self.classify_documents_task()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[self.ingest_task(), self.classify_documents_task(), self.extract_data_task()],
            process=Process.sequential,
            verbose=True,
            memory=False,
            max_rpm=max_rpm
        )

def build_crew():
    """Builds and returns the EmployeeVerificationApp crew"""
    return EmployeeVerificationApp().crew()

class ReadStructuredDataToolInput(BaseModel):
    """Input schema for ReadStructuredDataTool."""
    employee_id: str = Field(..., description="Employee ID to read structured data for")

class ReadStructuredDataTool(BaseTool):
    name: str = "read_structured_data"
    description: str = (
        "Read the extracted employee data from structured.json file. "
        "Use this tool to get the raw extracted data before validation. "
        "Input: employee_id (32-character hex string)"
    )
    args_schema: type[BaseModel] = ReadStructuredDataToolInput

    def _run(self, employee_id: str) -> str:
        """Read structured.json for the given employee_id."""
        from pathlib import Path
        
        try:
            root = Path(__file__).resolve().parents[2]
            data_file = root / "data_store" / employee_id / "structured.json"
            
            if not data_file.exists():
                return json.dumps({"error": f"structured.json not found for employee {employee_id}"})
            
            with data_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # If data contains large raw text chunks, remove them to prevent context overflow
            if "chunk_text" in data and isinstance(data["chunk_text"], list):
                # Remove chunk_text - it's raw OCR, not needed for validation
                del data["chunk_text"]
            
            json_str = json.dumps(data)
            
            # Hard limit: if still too large (>10K chars), truncate with warning
            if len(json_str) > 10000:
                return json.dumps({
                    "warning": "Data too large for context. Please check structured.json directly.",
                    "employee_id": employee_id,
                    "data_size": len(json_str),
                    "partial_data": json_str[:10000] + "...[truncated]"
                })
            
            return json_str
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class SaveValidatedDataToolInput(BaseModel):
    """Input schema for SaveValidatedDataTool."""
    validated_json: Union[str, dict] = Field(
        ..., 
        description="Validated employee data as JSON string or dictionary containing: name, address, date_of_birth, fathers_name, aadhar_no, pan_no, email, phone_no, education_details (array), employment_details (array)"
    )

class SaveValidatedDataTool(BaseTool):
    name: str = "save_validated_data"
    description: str = (
        "Save the final validated and corrected employee data to structured.json. "
        "Use this tool after you have validated and corrected all fields. "
        "Input can be a valid JSON string or dictionary containing: name, address, date_of_birth, fathers_name, "
        "aadhar_no, pan_no, email, phone_no, education_details (array), employment_details (array). "
        "This overwrites the initial extraction with your validated version."
    )
    args_schema: type[BaseModel] = SaveValidatedDataToolInput

    def _run(self, validated_json) -> str:
        """Save validated data to structured.json, overwriting vision tool's output."""
        import os
        from pathlib import Path
        
        try:
            # Handle both string and dict inputs
            if isinstance(validated_json, str):
                validated_data = json.loads(validated_json)
            elif isinstance(validated_json, dict):
                validated_data = validated_json
            else:
                return json.dumps({"error": f"Invalid input type: {type(validated_json)}", "success": False})
        except Exception as e:
            return json.dumps({"error": f"Invalid JSON provided: {str(e)}", "success": False})
        
        # Get employee_id from environment or streamer context
        employee_id = getattr(self, '_employee_id', None)
        if not employee_id:
            # Try to get from vision_extractions.json in recent data_store folders
            try:
                root = Path(__file__).resolve().parents[2]
                data_store = root / "data_store"
                if data_store.exists():
                    # Get most recent folder
                    folders = sorted([f for f in data_store.iterdir() if f.is_dir()], 
                                   key=lambda x: x.stat().st_mtime, reverse=True)
                    if folders:
                        employee_id = folders[0].name
            except Exception:
                pass
        
        if not employee_id:
            return json.dumps({"error": "Could not determine employee_id", "success": False})
        
        try:
            root = Path(__file__).resolve().parents[2]
            out_dir = root / "data_store" / employee_id
            out_dir.mkdir(parents=True, exist_ok=True)
            
            structured_path = out_dir / "structured.json"
            
            # Save the validated data
            with structured_path.open("w", encoding="utf-8") as f:
                json.dump(validated_data, f, ensure_ascii=False, indent=2)
            
            fields_populated = len([v for v in validated_data.values() if v is not None])
            
            return json.dumps({
                "success": True,
                "message": f"Validated data saved to structured.json",
                "employee_id": employee_id,
                "fields_populated": fields_populated,
                "total_fields": len(validated_data)
            })
            
        except Exception as e:
            return json.dumps({"error": str(e), "success": False})

class ValidateStructuredJSONTool(BaseTool):
    name: str = "validate_structured_json"
    description: str = (
        "Validate and normalize a JSON string for the required employee verification schema. "
        "Input must be the raw JSON (string). Output is a corrected JSON string with the new nested structure "
        "including education_details and employment_details as arrays."
    )

    required_keys: ClassVar[List[str]] = [
        "name", "address", "date_of_birth", "fathers_name", "aadhar_no", "pan_no", "email", "phone_no"
    ]
    
    education_keys: ClassVar[List[str]] = [
        "course_name", "institution_name", "course_tenure", "college_address", "percentage_obtained"
    ]
    
    employment_keys: ClassVar[List[str]] = [
        "employment_start_date", "employment_end_date", "company_name", "company_address", "date_of_joining"
    ]

    def _run(self, json_text: str) -> str:
        try:
            data = json.loads(json_text)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", json_text)
            if not m:
                return json.dumps({})
            try:
                data = json.loads(m.group(0))
            except Exception:
                return json.dumps({})

        if not isinstance(data, dict):
            return json.dumps({})

        out: Dict[str, object] = {}
        
        # Copy all keys from input data
        for k, v in data.items():
            # Skip complex nested structures for now, handle them separately if needed
            # or just pass them through. We'll pass through everything by default.
            out[k] = v

        # Apply specific cleaning to known fields if they exist
        
        # Clean basic string fields
        for k in self.required_keys:
            if k in out:
                v = out[k]
                if isinstance(v, str):
                    v = v.strip()
                    if v.lower() in {"", "null", "none", "na", "n/a"}:
                        v = None
                elif v not in (None,):
                    if isinstance(v, (list, dict)):
                        # Keep complex types as is for unknown fields, but for known basic fields, 
                        # if we expect a string and get a list/dict, it might be wrong.
                        # But for dynamic schema, let's be permissive.
                        pass 
                    else:
                        v = str(v)
                out[k] = v

        # Special handling for Aadhaar number
        if "aadhar_no" in out:
            v = out["aadhar_no"]
            if isinstance(v, str) and v:
                import re
                digits = re.sub(r"\D", "", v)

                def _verhoeff_check(num_str: str) -> bool:
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

                if len(digits) == 12 and _verhoeff_check(digits):
                    out["aadhar_no"] = f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"
                else:
                    # If invalid, keep original or set to None? 
                    # Let's set to None to indicate invalid extraction as per original logic
                    out["aadhar_no"] = None
        
        # Special handling for PAN number
        if "pan_no" in out:
            v = out["pan_no"]
            if isinstance(v, str) and v:
                import re
                v = v.upper().strip()
                # PAN format: AAAAA9999A (5 letters, 4 digits, 1 letter)
                if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", v):
                    out["pan_no"] = None
                else:
                    out["pan_no"] = v
        
        # Special handling for phone number
        if "phone_no" in out:
            v = out["phone_no"]
            if isinstance(v, str) and v:
                import re
                digits = re.sub(r"\D", "", v)
                # Indian phone: 10 digits starting with 6-9
                if len(digits) == 10 and digits[0] in "6789":
                    out["phone_no"] = digits
                else:
                    out["phone_no"] = None
        
        # Special handling for email
        if "email" in out:
            v = out["email"]
            if isinstance(v, str) and v:
                import re
                if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v):
                    out["email"] = None
        
        # Process education_details array (if present)
        if "education_details" in data and isinstance(data["education_details"], list):
            education_list = []
            for edu in data["education_details"]:
                if isinstance(edu, dict):
                    edu_obj = {}
                    # Copy all keys from edu object
                    for k, v in edu.items():
                        edu_obj[k] = v
                    
                    # Clean known keys
                    for k in self.education_keys:
                        if k in edu_obj:
                            v = edu_obj[k]
                            if isinstance(v, str):
                                v = v.strip()
                                if v.lower() in {"", "null", "none", "na", "n/a"}:
                                    v = None
                            elif v not in (None,):
                                v = str(v) if not isinstance(v, (list, dict)) else None
                            edu_obj[k] = v
                    
                    # Only add if at least one field is populated
                    if any(edu_obj.values()):
                        education_list.append(edu_obj)
            out["education_details"] = education_list
        
        # Process employment_details array (if present)
        if "employment_details" in data and isinstance(data["employment_details"], list):
            employment_list = []
            for emp in data["employment_details"]:
                if isinstance(emp, dict):
                    emp_obj = {}
                    # Copy all keys
                    for k, v in emp.items():
                        emp_obj[k] = v
                        
                    # Clean known keys
                    for k in self.employment_keys:
                        if k in emp_obj:
                            v = emp_obj[k]
                            if isinstance(v, str):
                                v = v.strip()
                                if v.lower() in {"", "null", "none", "na", "n/a"}:
                                    v = None
                            elif v not in (None,):
                                v = str(v) if not isinstance(v, (list, dict)) else None
                            emp_obj[k] = v
                    
                    # Only add if at least one field is populated
                    if any(emp_obj.values()):
                        employment_list.append(emp_obj)
            out["employment_details"] = employment_list
        
        return json.dumps(out)

