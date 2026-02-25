import os
import json
import time
import httpx
from typing import Dict, Any, Optional, Callable
from datetime import datetime


class FayeStreamer:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.faye_url = os.getenv("FAYE_URL", "https://faye.kriyam.ai/faye")
        self.faye_prefix = os.getenv("FAYE_CHANNEL_PREFIX", "").strip()
        self.faye_auth_token = os.getenv("FAYE_AUTH_TOKEN")
        
        if self.faye_prefix and not self.faye_prefix.startswith("/"):
            self.faye_prefix = "/" + self.faye_prefix
        self.channel = f"{self.faye_prefix.rstrip('/')}/{request_id}" if self.faye_prefix else f"/{request_id}"
    
    def send_update(self, message_type: str, content: Dict[str, Any], is_final: bool = False):
        if not self.faye_url:
            return
        
        standardized_content = {
            "step": content.get("step"),
            "agent": content.get("agent"),
            "tool": content.get("tool"),
            "input": content.get("input"),
            "status": content.get("status"),
            "is_final": is_final
        }
        
        filtered_content = {k: v for k, v in standardized_content.items() if v is not None}
        
        safe_additional_fields = [
            "message", "progress", "current_image", "total_images", "fields_extracted",
            "total_fields", "successful_extractions", "completion_rate", "populated_fields",
            "error", "success", "description", "task", "result_preview", "employee_data",
            "summary", "extracted_fields"
        ]
        
        for field in safe_additional_fields:
            if field in content and field not in filtered_content:
                value = content[field]
                if isinstance(value, str) and any(path_indicator in value.lower() for path_indicator in ['/home/', '/data_store/', 'path', 'directory', 'folder']):
                    value = self._clean_message_of_paths(value)
                filtered_content[field] = value
            
        payload = {
            "request_id": self.request_id,
            "timestamp": datetime.now().isoformat(),
            "type": message_type,
            "content": filtered_content,
            "is_final": is_final
        }
        
        message = {"channel": self.channel, "data": payload}
        if self.faye_auth_token:
            message["ext"] = {"authToken": self.faye_auth_token}
        
        try:
            with httpx.Client(timeout=10) as client:
                client.post(self.faye_url, json=[message])
                print(f"[Streaming] Sent {message_type} update to {self.channel}")
        except Exception as e:
            print(f"[Streaming] Failed to send update: {e}")
    
    def _clean_message_of_paths(self, message: str) -> str:
        import re
        message = re.sub(r'/[^\s]+/', '', message)
        message = re.sub(r'data_store/[a-f0-9]+', 'data store', message)
        message = re.sub(r'staged folder.*?for', 'files for', message)
        message = re.sub(r'expected_path.*', '', message)
        message = re.sub(r'root_path.*', '', message)
        message = re.sub(r'\s+', ' ', message).strip()
        return message


class CrewAIStreamingCallback:

    def __init__(self, streamer: FayeStreamer):
        self.streamer = streamer
        self.current_agent = None
        self.current_task = None
        self.step_counter = 0
        self.start_time = time.time()
    
    def on_task_start(self, task_name: str, task_description: str):
        self.current_task = task_name
        self.step_counter += 1
        
        self.streamer.send_update(
            "task_start",
            {
                "step": self.step_counter,
                "agent": "task_manager", 
                "task": task_name,
                "input": task_description,
                "status": "Starting task...",
                "description": task_description,
                "start_timestamp": self.start_time,
                "current_timestamp": time.time()
            }
        )
    
    def on_agent_action(self, agent_name: str, action: str, thought: str = None):
        self.current_agent = agent_name
        
        content = {
            "step": self.step_counter,
            "agent": agent_name,
            "input": action,
            "status": "Agent is working...",
            "action": action,
            "start_timestamp": self.start_time,
            "current_timestamp": time.time()
        }
        
        if thought:
            content["message"] = thought
            
        self.streamer.send_update("agent_action", content)
    
    def on_tool_use(self, tool_name: str, tool_input: str, agent_name: str = None):
        self.streamer.send_update(
            "tool_use",
            {
                "step": self.step_counter,
                "agent": agent_name or self.current_agent,
                "tool": tool_name,
                "input": tool_input[:200] + "..." if len(tool_input) > 200 else tool_input,
                "status": f"Using {tool_name}...",
                "start_timestamp": self.start_time,
                "current_timestamp": time.time()
            }
        )
    
    def on_tool_result(self, tool_name: str, result: str, success: bool = True):
        self.streamer.send_update(
            "tool_result",
            {
                "step": self.step_counter,
                "agent": self.current_agent,
                "tool": tool_name,
                "input": "Processing tool result",
                "status": f"{tool_name} completed {'successfully' if success else 'with error'}",
                "success": success,
                "result_preview": result[:300] + "..." if len(result) > 300 else result,
                "start_timestamp": self.start_time,
                "current_timestamp": time.time()
            }
        )
    
    def on_task_complete(self, task_name: str, result: str):
        self.streamer.send_update(
            "task_complete",
            {
                "step": self.step_counter,
                "agent": self.current_agent,
                "input": f"Completing task {task_name}",
                "status": f"Task {task_name} completed",
                "task": task_name,
                "result_preview": result[:300] + "..." if len(result) > 300 else result,
                "start_timestamp": self.start_time,
                "current_timestamp": time.time()
            }
        )
    
    def on_crew_complete(self, final_result: Any):
        try:
            import json
            from pathlib import Path
            
            root = Path(__file__).resolve().parents[2]
            structured_path = root / "data_store" / self.streamer.request_id / "structured.json"
            
            if structured_path.exists():
                with structured_path.open("r", encoding="utf-8") as f:
                    structured_data = json.load(f)
                
                self.streamer.send_update(
                    "verification_complete",
                    {
                        "step": self.step_counter + 1,
                        "agent": "verification_system",
                        "input": "Finalizing verification results",
                        "status": "Verification successfully completed",
                        "message": "All information has been extracted, analyzed, and verified.",
                        "structured_data": structured_data,
                        "fields_extracted": len([k for k, v in structured_data.items() if v is not None]),
                        "start_timestamp": self.start_time,
                        "end_timestamp": time.time()
                    },
                    is_final=True
                )
            else:
                self.streamer.send_update(
                    "crew_complete_fallback",
                    {
                        "step": self.step_counter + 1,
                        "agent": "verification_system",
                        "input": "Completing verification without structured data",
                        "status": "Processing complete",
                        "message": "Processing complete - no structured data file found",
                        "warning": "No structured data file found",
                        "start_timestamp": self.start_time,
                        "end_timestamp": time.time()
                    },
                    is_final=True
                )
        except Exception as e:
            self.streamer.send_update(
                "crew_error",
                {
                    "step": self.step_counter + 1,
                    "agent": "verification_system",
                    "input": "Processing final result with error",
                    "status": "Processing completed with error",
                    "error": f"Failed to load final result: {str(e)}",
                    "start_timestamp": self.start_time,
                    "end_timestamp": time.time()
                },
                is_final=True
            )


def create_streaming_wrapper(original_method: Callable, callback: CrewAIStreamingCallback):
    def wrapper(*args, **kwargs):
        callback.streamer.send_update(
            "crew_start",
            {
                "step": 0,
                "agent": "crew_manager",
                "input": "Starting employee verification process",
                "status": "Starting employee verification process...",
                "message": "Initializing agents and preparing to process documents",
                "start_timestamp": callback.start_time,
                "current_timestamp": time.time()
            }
        )
        
        try:
            # Execute the original method
            result = original_method(*args, **kwargs)
            
            # Send completion
            callback.on_crew_complete(result)
            return result
            
        except Exception as e:
            # Send error as final message
            callback.streamer.send_update(
                "crew_error",
                {
                    "step": callback.step_counter + 1,
                    "agent": "crew_manager",
                    "input": "Processing crew error",
                    "status": "Processing failed",
                    "error": str(e),
                    "start_timestamp": callback.start_time,
                    "end_timestamp": time.time()
                },
                is_final=True
            )
            raise
    
    return wrapper


class StreamingEmployeeVerificationApp:
    """Wrapper around EmployeeVerificationApp with streaming capabilities"""
    
    def __init__(self, request_id: str):
        from employee_verification_app.crew import EmployeeVerificationApp
        
        self.request_id = request_id
        self.streamer = FayeStreamer(request_id)
        self.callback = CrewAIStreamingCallback(self.streamer)
        self.base_app = EmployeeVerificationApp()
        
        # Inject streamer into tools so they can send updates
        self._inject_streamer_into_tools()
        
        # Get the crew instance
        self._crew = self.base_app.crew()
    
    def _inject_streamer_into_tools(self):
        """Inject the streamer into all tools so they can send updates"""
        # Get all agents from the crew
        ingestor = self.base_app.ingestor()
        classifier = self.base_app.classifier()
        vision_extractor = self.base_app.vision_extractor()
        
        # Inject streamer and employee_id into tools of each agent
        for agent in [ingestor, classifier, vision_extractor]:
            if hasattr(agent, 'tools') and agent.tools:
                for tool in agent.tools:
                    tool._streamer = self.streamer
                    tool._employee_id = self.request_id
    
    def kickoff_with_streaming(self, inputs: Dict[str, Any]):
        """Start the crew with streaming enabled"""
        
        # Send initial start message
        self.streamer.send_update(
            "crew_start",
            {
                "step": 0,
                "agent": "crew_manager",
                "input": "Starting employee verification process",
                "status": "Starting employee verification process...",
                "message": "Initializing agents and preparing to process documents",
                "start_timestamp": self.callback.start_time,
                "current_timestamp": time.time()
            }
        )
        
        # Send detailed progress updates
        self.callback.on_task_start(
            "ingest_task", 
            "Ingesting and organizing uploaded employee documents"
        )
        
        # Simulate some thinking steps for the ingestion phase
        self.callback.on_agent_action(
            "ingestor", 
            "analyzing_files", 
            "Examining uploaded files and determining processing strategy"
        )
        
        self.callback.on_tool_use(
            "IngestEmployeeFilesTool", 
            f"Processing files for employee {inputs.get('employee_id')}", 
            "ingestor"
        )
        
        # Continue with vision extraction
        self.callback.on_task_start(
            "extract_data_task",
            "Analyzing documents using advanced vision AI to extract structured information"
        )
        
        self.callback.on_agent_action(
            "vision_extractor",
            "preparing_vision_analysis", 
            "Preparing documents for vision-based analysis and field extraction"
        )
        
        self.callback.on_tool_use(
            "ExtractStructuredDataFromImagesTool",
            f"Analyzing images for employee {inputs.get('employee_id')}",
            "vision_extractor"
        )
        
        # Execute the actual crew with proper error handling
        try:
            result = self._crew.kickoff(inputs=inputs)
            
            # Debug: Check what type of result we got
            print(f"[StreamingApp] Crew result type: {type(result)}")
            print(f"[StreamingApp] Crew result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            # Handle different result types
            result_str = None
            
            # Try different ways to extract the actual result
            if hasattr(result, 'tasks_output') and result.tasks_output:
                # Get the last task output (extract_data_task)
                tasks_output = result.tasks_output
                print(f"[StreamingApp] Found tasks_output: {len(tasks_output)} tasks")
                
                if len(tasks_output) > 0:
                    last_task_output = tasks_output[-1]
                    print(f"[StreamingApp] Last task output type: {type(last_task_output)}")
                    print(f"[StreamingApp] Last task attributes: {[a for a in dir(last_task_output) if not a.startswith('_')]}")
                    
                    # Try to get the output - Pydantic model first
                    # if hasattr(last_task_output, 'pydantic') and last_task_output.pydantic:
                    #     # Convert Pydantic model to dict then to JSON string
                    #     result_str = last_task_output.pydantic.model_dump_json()
                    #     print(f"[StreamingApp] Using last_task.pydantic (converted to JSON): {result_str[:200]}")
                    if hasattr(last_task_output, 'json_dict') and last_task_output.json_dict:
                        import json
                        result_str = json.dumps(last_task_output.json_dict)
                        print(f"[StreamingApp] Using last_task.json_dict: {result_str[:200]}")
                    elif hasattr(last_task_output, 'raw') and last_task_output.raw:
                        result_str = last_task_output.raw
                        print(f"[StreamingApp] Using last_task.raw: {str(result_str)[:200]}")
                    elif hasattr(last_task_output, 'output') and last_task_output.output:
                        result_str = last_task_output.output
                        print(f"[StreamingApp] Using last_task.output: {str(result_str)[:200]}")
                    
                    # If output contains the tool result, try to parse it
                    if result_str and isinstance(result_str, str):
                        # Check if it's already JSON
                        import json
                        try:
                            json.loads(result_str)
                            print("[StreamingApp] Result is valid JSON")
                        except Exception:
                            # Try to extract JSON from tool output markers
                            import re
                            json_match = re.search(r'\{[\s\S]*\}', result_str)
                            if json_match:
                                potential_json = json_match.group(0)
                                try:
                                    json.loads(potential_json)
                                    result_str = potential_json
                                    print(f"[StreamingApp] Extracted JSON from response: {result_str[:200]}")
                                except:
                                    print(f"[StreamingApp] Could not parse as JSON: {result_str[:200]}")
            
            if not result_str and hasattr(result, 'raw'):
                result_str = result.raw
                print(f"[StreamingApp] Using result.raw: {str(result_str)[:200] if result_str else 'None'}")
            
            if not result_str and hasattr(result, 'json_dict'):
                import json
                result_str = json.dumps(result.json_dict)
                print(f"[StreamingApp] Using result.json_dict: {result_str[:200]}")
            
            if not result_str and hasattr(result, 'output'):
                result_str = result.output
                print(f"[StreamingApp] Using result.output: {str(result_str)[:200] if result_str else 'None'}")
            
            if not result_str:
                result_str = str(result)
                print(f"[StreamingApp] Using str(result): {result_str[:200]}")
            
            print(f"[StreamingApp] Final result_str: {result_str[:300] if result_str else 'None'}")
            
            self.callback.on_crew_complete(result_str or result)
        except Exception as e:
            print(f"[StreamingApp] Crew execution error: {e}")
            import traceback
            traceback.print_exc()
            
            self.streamer.send_update(
                "crew_error",
                {
                    "step": self.callback.step_counter + 1,
                    "agent": "crew_manager",
                    "input": "Handling crew execution error",
                    "status": "Processing failed",
                    "error": str(e),
                    "start_timestamp": self.callback.start_time,
                    "end_timestamp": time.time()
                },
                is_final=True
            )
            raise
        
        try:
            import json
            from pathlib import Path
            
            root = Path(__file__).resolve().parents[2] 
            structured_path = root / "data_store" / self.request_id / "structured.json"
            
            if structured_path.exists():
                with structured_path.open("r", encoding="utf-8") as f:
                    structured_data = json.load(f)
                
                # Send final verification results
                self.streamer.send_update(
                    "final_verification_result",
                    {
                        "step": self.callback.step_counter + 1,
                        "agent": "verification_system",
                        "input": "Finalizing verification results",
                        "status": "Verification processing complete",
                        "employee_data": structured_data,
                        "summary": {
                            "total_fields": len(structured_data),
                            "populated_fields": len([k for k, v in structured_data.items() if v is not None]),
                            "completion_rate": f"{len([k for k, v in structured_data.items() if v is not None]) / len(structured_data) * 100:.1f}%"
                        },
                        "start_timestamp": self.callback.start_time,
                        "end_timestamp": time.time()
                    },
                    is_final=True
                )
        except Exception as e:
            print(f"[Streaming] Failed to send final structured data: {e}")
        
        if 'result_str' in locals() and result_str:
            return result_str
        return result