import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from employee_verification_app.crew import EmployeeVerificationApp

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    default_employee_id = os.getenv("EMP_ID", "2160550")
    workspace_root = Path(__file__).resolve().parents[2]
    default_source = Path(os.getenv("SOURCE_FOLDER", str(workspace_root / "test_dataset" / default_employee_id)))
    inputs = {
        "employee_id": default_employee_id,
        "source_folder": str(default_source),
        "current_year": str(datetime.now().year),
    }
    try:
        print(f"Starting crew for employee {default_employee_id} with vision extraction...")
        EmployeeVerificationApp().crew().kickoff(inputs=inputs)

        # Check if structured.json was created by crew
        structured_path = workspace_root / "data_store" / default_employee_id / "structured.json"
        if structured_path.exists():
            print(f"✓ Extraction completed: {structured_path}")
            # Print summary
            import json as _json
            with structured_path.open("r") as f:
                data = _json.load(f)
                fields_found = len([k for k, v in data.items() if v is not None])
                print(f"✓ Fields extracted: {fields_found}")
        else:
            print("⚠ Warning: structured.json not found, attempting fallback...")
            return 
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def train():
    inputs = {"employee_id": "2160550", "source_folder": "./test_dataset/2160550", 'current_year': str(datetime.now().year)}
    try:
        EmployeeVerificationApp().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    try:
        EmployeeVerificationApp().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    inputs = {"employee_id": "2160550", "source_folder": "./test_dataset/2160550", "current_year": str(datetime.now().year)}
    try:
        EmployeeVerificationApp().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
