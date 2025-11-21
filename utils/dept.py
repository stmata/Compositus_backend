import re

DEPT_ALIASES = {
"eap_professional_interviews": "EAP_EntretiensProfessionnels",
"eap_professors": "EAP_Professeurs",
"eap_administratif": "EAP_Administratif",
"eap_entretiensprofessionnels": "EAP_EntretiensProfessionnels",
"eap_professeurs": "EAP_Professeurs",
"eap_administratif": "EAP_Administratif",
}

def _dept_alias_key(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "_").replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "", s)

_def_all = {"", "all", "tout", "tous", "toutes", "everything", "*", "eap_all"}

def is_all(dept: str) -> bool:
    return _dept_alias_key(dept) in _def_all

def normalize_dept(s: str) -> str:
    key = _dept_alias_key(s)
    return DEPT_ALIASES.get(key, s)