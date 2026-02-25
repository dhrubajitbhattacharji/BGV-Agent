# New merge logic for nested structure
# This will be integrated into vision_tool.py

def merge_extractions_to_nested_structure(all_extractions, classifications, logger):
    """
    Merge all extractions into the new nested structure format.
    Returns dict with basic fields + education_details[] + employment_details[]
    """
    
    # Define field priorities
    field_priorities = {
        "aadhar_no": ["AADHAR_CARD", "APPLICATION_FORM"],
        "pan_no": ["PAN_CARD", "APPLICATION_FORM"],
        "name": ["AADHAR_CARD", "PAN_CARD", "RESUME", "DEGREE_CERTIFICATE", "EXPERIENCE_CERTIFICATE", "MARKSHEET", "APPLICATION_FORM", "AFFIDAVIT", "OTHER"],
        "fathers_name": ["AADHAR_CARD", "PAN_CARD", "MARKSHEET", "DEGREE_CERTIFICATE", "APPLICATION_FORM", "AFFIDAVIT", "RESUME", "OTHER"],
        "address": ["AADHAR_CARD", "APPLICATION_FORM", "RESUME", "OTHER"],
        "date_of_birth": ["AADHAR_CARD", "APPLICATION_FORM", "RESUME", "OTHER"],
        "email": ["RESUME", "APPLICATION_FORM", "OTHER"],
        "phone_no": ["RESUME", "APPLICATION_FORM", "OTHER"],
    }
    
    basic_fields = ["name", "address", "date_of_birth", "fathers_name", "aadhar_no", "pan_no", "email", "phone_no"]
    
    # Helper to detect formal education
    def _is_formal_education(course_name, tenure):
        if not course_name or not isinstance(course_name, str):
            return False
        
        course_lower = course_name.lower()
        formal_keywords = [
            "matriculation", "ssc", "secondary", "intermediate", "hsc", "higher secondary",
            "diploma", "graduation", "b.tech", "b.e", "b.sc", "b.com", "b.a", "bachelor",
            "m.tech", "m.e", "m.sc", "m.com", "m.a", "master", "phd", "doctorate"
        ]
        short_course_keywords = [
            "typing", "certificate course", "workshop", "training", "photography",
            "computer course", "tally", "spoken english"
        ]
        
        if any(keyword in course_lower for keyword in short_course_keywords):
            return False
        if any(keyword in course_lower for keyword in formal_keywords):
            return True
        
        # Check tenure
        if tenure and isinstance(tenure, str):
            import re
            year_match = re.findall(r"(\d{4})", tenure)
            if len(year_match) >= 2:
                try:
                    duration = int(year_match[-1]) - int(year_match[0])
                    if duration >= 2:
                        return True
                except:
                    pass
        return False
    
    # Merge basic fields
    merged = {}
    merge_sources = {}
    
    for field in basic_fields:
        merged[field] = None
        merge_sources[field] = None
        
        priority_order = field_priorities.get(field, ["OTHER"])
        best_value = None
        best_priority = len(priority_order)
        best_doc_type = None
        best_file = None
        
        for ext in all_extractions:
            if "data" in ext and isinstance(ext["data"], dict):
                val = ext["data"].get(field)
                doc_type = ext.get("doc_type", "OTHER")
                
                if val is not None and val != "null" and str(val).strip():
                    try:
                        current_priority = priority_order.index(doc_type)
                    except ValueError:
                        current_priority = len(priority_order)
                    
                    if best_value is None or current_priority < best_priority:
                        best_value = val
                        best_priority = current_priority
                        best_doc_type = doc_type
                        best_file = ext.get("file", "unknown")
        
        if best_value is not None:
            # Apply validation for specific fields
            if field == "aadhar_no":
                from . import vision_tool
                validated = vision_tool._validate_and_format_aadhaar(best_value)
                if validated:
                    merged[field] = validated
                    merge_sources[field] = {"doc_type": best_doc_type, "file": best_file, "priority": best_priority}
                else:
                    # Try alternatives
                    for ext in all_extractions:
                        if "data" in ext and isinstance(ext["data"], dict):
                            alt_val = ext["data"].get(field)
                            if alt_val and alt_val != best_value:
                                validated_alt = vision_tool._validate_and_format_aadhaar(alt_val)
                                if validated_alt:
                                    merged[field] = validated_alt
                                    merge_sources[field] = {"doc_type": ext.get("doc_type"), "file": ext.get("file"), "priority": -1, "note": "Fallback"}
                                    break
            else:
                merged[field] = best_value
                merge_sources[field] = {"doc_type": best_doc_type, "file": best_file, "priority": best_priority}
    
    # Collect education records
    education_list = []
    education_doc_types = ["MARKSHEET", "DEGREE_CERTIFICATE"]
    
    for ext in all_extractions:
        if "data" in ext and isinstance(ext["data"], dict):
            doc_type = ext.get("doc_type", "OTHER")
            data = ext["data"]
            
            # Check if this has education fields
            course_name = data.get("course_name") or data.get("last_course_name")
            institution_name = data.get("college_name") or data.get("last_college_name")
            
            if course_name and _is_formal_education(course_name, data.get("course_tenure")):
                education_list.append({
                    "course_name": course_name,
                    "college_name": institution_name,
                    "course_tenure": data.get("course_tenure"),
                    "college_address": data.get("college_address"),
                    "percentage_obtained": data.get("percentage_obtained"),
                    "total_marks_obtained": data.get("total_marks_obtained")
                })
    
    # Collect employment records
    employment_list = []
    employment_doc_types = ["EXPERIENCE_CERTIFICATE", "SERVICE_LETTER"]
    
    for ext in all_extractions:
        if "data" in ext and isinstance(ext["data"], dict):
            doc_type = ext.get("doc_type", "OTHER")
            data = ext["data"]
            
            # Check if this has employment fields
            company_name = data.get("company_name")
            
            if company_name:
                # Normalize for comparison
                company_norm = str(company_name).lower().strip()
                
                # Check if we already have a record for this company
                existing_record = None
                for record in employment_list:
                    if record["company_name"] and str(record["company_name"]).lower().strip() == company_norm:
                        existing_record = record
                        break
                
                if existing_record:
                    # Merge data: prefer non-null/longer values
                    if not existing_record.get("employment_start_date") and data.get("employment_start_date"):
                        existing_record["employment_start_date"] = data.get("employment_start_date")
                    
                    if not existing_record.get("employment_end_date") and data.get("employment_end_date"):
                        existing_record["employment_end_date"] = data.get("employment_end_date")
                        
                    # Prefer longer address
                    curr_addr = existing_record.get("company_address") or ""
                    new_addr = data.get("company_address") or ""
                    if len(new_addr) > len(curr_addr):
                        existing_record["company_address"] = new_addr
                else:
                    # Add new record (excluding date_of_joining)
                    # STRICT FILTER: Must have employment_start_date
                    start_date = data.get("employment_start_date")
                    if start_date:
                        employment_list.append({
                            "employment_start_date": start_date,
                            "employment_end_date": data.get("employment_end_date"),
                            "company_name": company_name,
                            "company_address": data.get("company_address")
                        })
    
    merged["education_details"] = education_list
    merged["employment_details"] = employment_list
    
    return merged, merge_sources
