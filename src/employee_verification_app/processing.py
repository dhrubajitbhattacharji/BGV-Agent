import json
import re
from pathlib import Path
from typing import Dict, Any, List

from employee_verification_app.tools.custom_tool import ExtractTextForEmployeeTool

# --- helpers (anchor-based extraction and normalization) ---

STATE_NAMES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Delhi","Goa","Gujarat", "Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka","Kerala",
    "Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha", "Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh", "Uttarakhand","West Bengal","Puducherry"
]
STATE_REGEX = re.compile(r"\b(?:" + "|".join(map(re.escape, STATE_NAMES)) + r")\b", re.IGNORECASE)
PIN_REGEX = re.compile(r"\b\d{6}\b")
AADHAR_REGEX = re.compile(r"\b(\d{4})[ \-]?(\d{4})[ \-]?(\d{4})\b")
DATE_REGEX = re.compile(r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b")

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _extract_after_anchor(lines: List[str], anchors: List[str], max_lines_after: int = 3, max_chars: int = 200):
    # returns (value, pattern) or (None, None)
    for i, ln in enumerate(lines):
        for a in anchors:
            if re.search(a, ln, re.IGNORECASE):
                # prefer right side of ':' or '-' on same line
                m = re.split(r"[:\-]\s*", ln, maxsplit=1)
                if len(m) > 1 and _norm_space(m[1]):
                    val = _norm_space(m[1])
                    return val, a
                # else use next non-empty lines up to cap
                buf, used = [], 0
                for j in range(i + 1, min(len(lines), i + 1 + max_lines_after)):
                    part = _norm_space(lines[j])
                    if not part:
                        break
                    buf.append(part)
                    used += len(part) + 1
                    if used >= max_chars:
                        break
                if buf:
                    return _norm_space(" ".join(buf)), a
    return None, None

def _extract_name(lines: List[str], texts: str):
    val, pat = _extract_after_anchor(lines, [
        r"\bname\b", r"\bcandidate\s*name\b", r"\bemployee\s*name\b", r"\bstudent\s*name\b"
    ], max_lines_after=2)
    if not val:
        # fallback: first line that looks like a person name (2-4 capitalized words)
        for ln in lines:
            tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z'.-]+", ln)]
            if 2 <= len(tokens) <= 4 and all(t[0].isalpha() for t in tokens):
                val, pat = _norm_space(" ".join(tokens)).title(), "fallback_name_line"
                break
    if val:
        val = _norm_space(re.sub(r"^(mr|mrs|ms|shri|smt)\.?\s+", "", val, flags=re.IGNORECASE)).title()
    return val, pat

def _extract_father(lines: List[str]):
    val, pat = _extract_after_anchor(lines, [
        r"\bfather'?s?\s*name\b", r"\bs\/o\b", r"\bson\s+of\b", r"\bparent'?s?\s*name\b"
    ], max_lines_after=2)
    if val:
        val = _norm_space(re.sub(r"^(mr|shri)\.?\s+", "", val, flags=re.IGNORECASE)).title()
    return val, pat

def _extract_dob(texts: str, lines: List[str]):
    # prefer anchored DOB
    v, p = _extract_after_anchor(lines, [r"\bdob\b", r"\bdate\s*of\s*birth\b"], max_lines_after=1)
    if v:
        m = DATE_REGEX.search(v)
        if m:
            d, mth, y = m.groups()
            y = y if len(y) == 4 else ("20" + y if int(y) < 50 else "19" + y)
            return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}", "DOB(anchor)"
    # generic
    m = DATE_REGEX.search(texts)
    if m:
        d, mth, y = m.groups()
        y = y if len(y) == 4 else ("20" + y if int(y) < 50 else "19" + y)
        return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}", r"DATE_REGEX"
    return None, None

def _extract_aadhar(texts: str):
    m = AADHAR_REGEX.search(texts)
    if not m:
        return None, None
    val = "-".join(m.groups())
    return val, r"AADHAR_REGEX"

def _extract_address_city_state(lines: List[str], texts: str):
    address, addr_pat = _extract_after_anchor(lines, [
        r"\baddress\b", r"\bperm(?:anent)?\s*address\b", r"\bpresent\s*address\b", r"\bresidence\b"
    ], max_lines_after=4, max_chars=240)

    # Fallback: capture block around PIN
    if not address:
        pin_m = PIN_REGEX.search(texts)
        if pin_m:
            pin_idx = pin_m.start()
            start = max(0, pin_idx - 180)
            block = _norm_space(texts[start:pin_idx + 6])
            address, addr_pat = block, "address(pin_block)"

    state = None
    city = None
    if address:
        sm = STATE_REGEX.search(address)
        if sm:
            state = sm.group(0).title()
        # city heuristic: token immediately before state or before PIN
        if state:
            before_state = address[:sm.start()].strip().rstrip(",")
            tokens = [t for t in re.split(r"[,\s]+", before_state) if t]
            if tokens:
                city = tokens[-1].title()
        if not city:
            pin_m = PIN_REGEX.search(address)
            if pin_m:
                left = address[:pin_m.start()].strip().rstrip(",")
                tokens = [t for t in re.split(r"[,\s]+", left) if t]
                if tokens:
                    city = tokens[-1].title()
        # normalize
        address = _norm_space(address)
    return address, addr_pat, city, state

def _extract_course(texts: str):
    # Detect degree + optional specialization
    m = re.search(r"\b(B\.?\s*Tech|B\.?\s*E|Bachelor\s+of\s+Technology|Bachelor\s+of\s+Engineering)\b", texts, re.IGNORECASE)
    spec = None
    if m:
        spec_m = re.search(r"(?:\(|in\s+)([A-Za-z &\/]{2,40})(?:\)|\b)", texts[m.end():], re.IGNORECASE)
        spec = _norm_space(spec_m.group(1)) if spec_m else None
        degree = re.sub(r"\s+", "", m.group(0)).replace("Bachelorof", "Bachelor of ").replace("B.Tech", "B.Tech").replace("B.E", "B.E")
        pretty = "B.Tech" if "tech" in degree.lower() else "B.E"
        return f"{pretty} ({spec})" if spec else pretty, "course(degree/spec)"
    return None, None

def _extract_tenure(texts: str):
    years = re.findall(r"(20\d{2})\s*[-/–]\s*(?:20)?(\d{2})", texts)
    if years:
        yrs = [(int(a), int(b)) for a, b in years]
        start = min(a for a, _ in yrs)
        end_suf = max(b for _, b in yrs)
        end = 2000 + end_suf if end_suf < 100 else end_suf
        if end >= start:
            return f"{start}-{end}", "tenure(year-range)"
    return None, None

def _extract_college(lines: List[str], texts: str):
    # Anchor first
    val, pat = _extract_after_anchor(lines, [r"\bcollege\/?institution\b", r"\bcollege\s*name\b", r"\binstitution\b"], max_lines_after=2)
    if val:
        return val.upper(), "college(anchor)"
    # Heuristic: longest uppercase line with keywords
    candidates = []
    for ln in lines:
        if re.search(r"\b(UNIVERSITY|INSTITUTE|COLLEGE|POLYTECHNIC|INSTITUTION)\b", ln, re.IGNORECASE):
            tokens = _norm_space(ln)
            # prefer mostly uppercase lines
            if sum(ch.isupper() for ch in tokens if ch.isalpha()) >= max(1, int(0.6 * sum(ch.isalpha() for ch in tokens))):
                candidates.append(tokens.upper())
    if candidates:
        return sorted(candidates, key=len, reverse=True)[0], "college(uppercase-line)"
    return None, None

def build_structured_output_with_trace(employee_id: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    trace: Dict[str, Any] = {"employee_id": employee_id}
    try:
        raw = ExtractTextForEmployeeTool()._run(employee_id)
        arr: List[Dict[str, Any]] = json.loads(raw) if isinstance(raw, str) else raw  # type: ignore
    except Exception as e:
        trace["ocr_error"] = str(e)
        arr = []  # type: ignore

    texts = "\n".join([str(it.get("text", "")) for it in arr])
    trace["ocr_files"] = [it.get("filename") for it in arr]
    trace["ocr_total_chars"] = len(texts)

    field_matches: Dict[str, Any] = {}

    lines = [ln.strip() for ln in texts.splitlines()]

    # Name
    name, pat = _extract_name(lines, texts)
    if name:
        field_matches["name"] = {"pattern": pat, "value": name}

    # Father's name
    fathers_name, pat = _extract_father(lines)
    if fathers_name:
        field_matches["fathers_name"] = {"pattern": pat, "value": fathers_name}

    # DOB
    dob, pat = _extract_dob(texts, lines)
    if dob:
        field_matches["date_of_birth"] = {"pattern": pat, "value": dob}

    # Aadhar
    aadhar, pat = _extract_aadhar(texts)
    if aadhar:
        field_matches["aadhar_no"] = {"pattern": pat, "value": aadhar}

    # Address, City, State
    address, addr_pat, city, state = _extract_address_city_state(lines, texts)
    if address:
        field_matches["address"] = {"pattern": addr_pat, "value": address}
    if city:
        field_matches["city"] = {"pattern": "city(from-address)", "value": city}
    if state:
        field_matches["state"] = {"pattern": "state(from-address)", "value": state}

    # Course and College
    last_course_name, pat = _extract_course(texts)
    if last_course_name:
        field_matches["last_course_name"] = {"pattern": pat, "value": last_course_name}

    last_college_name, pat = _extract_college(lines, texts)
    if last_college_name:
        field_matches["last_college_name"] = {"pattern": pat, "value": last_college_name}

    # Tenure
    course_tenure, pat = _extract_tenure(texts)
    if course_tenure:
        field_matches["course_tenure"] = {"pattern": pat, "value": course_tenure}

    out = {
        "name": name,
        "address": address,
        "date_of_birth": dob,
        "fathers_name": fathers_name,
        "last_course_name": last_course_name,
        "last_college_name": last_college_name,
        "course_tenure": course_tenure,
        "college_address": None,
        "city": city,
        "state": state,
        "aadhar_no": aadhar,
        "employment_start_date": None,
        "employment_end_date": None,
        "company_name": None,
        "company_address": None,
        "date_of_joining": None,
    }

    # persist
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "data_store" / employee_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "structured.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    trace["extraction_matches"] = field_matches
    trace["output_json_path"] = str((out_dir / "structured.json").resolve())

    return out, trace


def build_structured_output(employee_id: str) -> Dict[str, Any]:
    data, _ = build_structured_output_with_trace(employee_id)
    return data