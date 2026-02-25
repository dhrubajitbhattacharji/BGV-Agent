"""
Markdown formatter for employee verification data.
Converts structured JSON to readable markdown format with dynamic field detection.
"""

import json
from typing import Dict, Any, List, Union


def _format_key_as_label(key: str) -> str:
    """Convert snake_case key to Title Case Label."""
    return " ".join(word.capitalize() for word in key.replace("_", " ").split())


def _format_value(value: Any, indent: int = 0) -> str:
    """Format a value for markdown display."""
    prefix = "  " * indent
    
    if value is None:
        return "_Not provided_"
    elif isinstance(value, bool):
        return "Yes" if value else "No"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return value if value.strip() else "_Not provided_"
    elif isinstance(value, dict):
        # Format nested dict as indented key-value pairs
        lines = []
        for k, v in value.items():
            if v is not None and v != "":
                label = _format_key_as_label(k)
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}- **{label}:**")
                    lines.append(_format_value(v, indent + 1))
                else:
                    lines.append(f"{prefix}- **{label}:** {_format_value(v, 0)}")
        return "\n".join(lines) if lines else "_Not provided_"
    elif isinstance(value, list):
        if not value:
            return "_Not provided_"
        # Check if it's a list of primitives or objects
        if all(isinstance(item, (str, int, float, bool)) for item in value):
            return ", ".join(str(item) for item in value if item)
        else:
            # List of objects - will be handled by caller
            return None
    else:
        return str(value)


def format_employee_data_as_markdown(data: Dict[str, Any]) -> str:
    """
    Dynamically format employee verification data as markdown based on JSON structure.
    
    Args:
        data: Dict containing employee data with any structure
        
    Returns:
        Markdown formatted string
    """
    if not data or not isinstance(data, dict):
        return "# Employee Verification Report\n\nNo data available."
    
    # Handle error cases
    if "error" in data:
        return f"# Employee Verification Report\n\n**Error:** {data['error']}"
    
    md_lines = []
    md_lines.append("# Employee Verification Report")
    md_lines.append("")
    
    # Define field categories for organization
    personal_info_keys = {
        "name", "father_name", "fathers_name", "mother_name", "mothers_name",
        "date_of_birth", "dob", "gender", "marital_status", "nationality",
        "blood_group", "religion", "caste"
    }
    
    identity_keys = {
        "aadhaar_number", "aadhar_no", "pan_number", "pan_no", 
        "passport_number", "driving_license", "voter_id"
    }
    
    contact_keys = {
        "mobile_number", "phone_no", "phone", "email", "address",
        "current_address", "permanent_address", "contact_number"
    }
    
    array_keys = {
        "education_details", "education", "educational_qualifications",
        "employment_details", "employment", "work_experience", "experience",
        "certifications", "skills", "languages", "projects", "references"
    }
    
    # Track which keys we've already processed
    processed_keys = set()
    
    # Section 1: Personal Information
    personal_section = []
    for key, value in data.items():
        if key.lower() in personal_info_keys and value is not None and value != "":
            label = _format_key_as_label(key)
            formatted_value = _format_value(value)
            if formatted_value and formatted_value != "_Not provided_":
                personal_section.append(f"**{label}:** {formatted_value}")
                processed_keys.add(key)
    
    if personal_section:
        md_lines.append("## Personal Information")
        md_lines.append("")
        md_lines.extend(personal_section)
        md_lines.append("")
    
    # Section 2: Identity Documents
    identity_section = []
    for key, value in data.items():
        if key.lower() in identity_keys and value is not None and value != "":
            label = _format_key_as_label(key)
            formatted_value = _format_value(value)
            if formatted_value and formatted_value != "_Not provided_":
                identity_section.append(f"**{label}:** {formatted_value}")
                processed_keys.add(key)
    
    if identity_section:
        md_lines.append("## Identity Documents")
        md_lines.append("")
        md_lines.extend(identity_section)
        md_lines.append("")
    
    # Section 3: Contact Information
    contact_section = []
    for key, value in data.items():
        if key.lower() in contact_keys and value is not None and value != "":
            label = _format_key_as_label(key)
            formatted_value = _format_value(value)
            if formatted_value and formatted_value != "_Not provided_":
                contact_section.append(f"**{label}:** {formatted_value}")
                processed_keys.add(key)
    
    if contact_section:
        md_lines.append("## Contact Information")
        md_lines.append("")
        md_lines.extend(contact_section)
        md_lines.append("")
    
    # Section 4: Dynamic Array Sections (Education, Employment, etc.)
    for key, value in data.items():
        if key.lower() in array_keys and isinstance(value, list) and value:
            processed_keys.add(key)
            
            # Create section header from key name
            section_name = _format_key_as_label(key)
            md_lines.append(f"## {section_name}")
            md_lines.append("")
            
            # Process each item in the array
            for idx, item in enumerate(value, 1):
                if isinstance(item, dict):
                    md_lines.append(f"### {section_name[:-1] if section_name.endswith('s') else section_name} {idx}")
                    md_lines.append("")
                    
                    # Dynamically add all fields from the object
                    for item_key, item_value in item.items():
                        if item_value is not None and item_value != "":
                            item_label = _format_key_as_label(item_key)
                            formatted_value = _format_value(item_value)
                            if formatted_value and formatted_value != "_Not provided_":
                                md_lines.append(f"**{item_label}:** {formatted_value}")
                    
                    md_lines.append("")
                elif isinstance(item, str):
                    md_lines.append(f"- {item}")
            
            md_lines.append("")
    
    # Section 5: Other Fields (catch-all for remaining fields)
    other_section = []
    for key, value in data.items():
        if key not in processed_keys and value is not None and value != "":
            # Skip internal/metadata fields
            if key.startswith("_") or key in ["parse_error", "raw_response", "success"]:
                continue
                
            label = _format_key_as_label(key)
            
            # Handle complex values
            if isinstance(value, list):
                if not value:
                    continue
                other_section.append(f"**{label}:**")
                for item in value:
                    if isinstance(item, dict):
                        other_section.append(_format_value(item, 1))
                    else:
                        other_section.append(f"  - {item}")
            elif isinstance(value, dict):
                other_section.append(f"**{label}:**")
                other_section.append(_format_value(value, 1))
            else:
                formatted_value = _format_value(value)
                if formatted_value and formatted_value != "_Not provided_":
                    other_section.append(f"**{label}:** {formatted_value}")
    
    if other_section:
        md_lines.append("## Additional Information")
        md_lines.append("")
        md_lines.extend(other_section)
        md_lines.append("")
    
    # Footer
    md_lines.append("---")
    md_lines.append("*This report was automatically generated from employee verification documents.*")
    
    return "\n".join(md_lines)


def save_markdown_report(data: Dict[str, Any], output_path: str) -> bool:
    """
    Generate and save markdown report to file.
    
    Args:
        data: Employee data dict
        output_path: Path to save markdown file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        markdown_content = format_employee_data_as_markdown(data)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        return True
    except Exception as e:
        print(f"Error saving markdown report: {e}")
        return False
