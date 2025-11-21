from __future__ import annotations
from typing import Dict, Any, List
import json

def build_summary_prompt(user_data: dict) -> str:
    """
    Build a robust summarization prompt for HR talent matching.
    Focus: last 5 years of evaluation data.
    Output: JSON with identity, long summary, signals, goals, assessments, evidence, decision features.
    """

    return f"""
You are an expert HR summarizer. You read structured multi-year evaluations of one collaborator.
Data can be in French or English, partially missing, or inconsistent.

Objective:
Produce a detailed, decision-ready profile for job matching. 
This must help HR distinguish between two equally strong candidates.

Rules:
- Focus on the last 5 years. Older data only if it shows unique trends.
- Include as much useful information as possible. No word cap.
- Remove duplicates. Never invent missing values.
- Normalize numbers (hours, days, %, points). Keep CNRS ranks (1/2/3/4).
- Extract concrete signals: expertise topics, teaching (audiences, hours, evals), research (outputs, ranks, roles), leadership & management, institutional impact, validations/raises, availability, mobility, goals, training needs, risks.
- Style: neutral, factual, professional. Multi-paragraph summary, not bullet points.

Input:
Collaborator data:
{user_data}

Output:
Return a single JSON object with this schema:

{{
  "identity": {{
    "name": "",
    "matricule": "",
    "current_role": "",
    "current_profile": "",
    "qualification": ""
  }},
  "summary_long": "<multi-paragraph, detailed synthesis of the last 5 years>",
  "signals": {{
    "expertise_topics": ["..."],
    "teaching": {{
      "audiences": ["..."],
      "eval_satisfaction_pct_recent": number|null,
      "hours_recent": number|null,
      "notes": ""
    }},
    "research": {{
      "profile_level": "A|B|C|D",
      "pubs_rank2plus_since_YYYY": number|null,
      "points_CI_recent": number|null,
      "roles": ["..."],
      "notes": ""
    }},
    "leadership": {{
      "roles_recent": ["..."],
      "responsibility_days_recent": number|null,
      "institutional_impact": ["..."]
    }},
    "availability": {{
      "teaching_hours": number|null,
      "encadrement_days": number|null,
      "contrib_intellect_days": number|null,
      "responsibility_days": number|null
    }},
    "mobility": {{ "multicampus": "Yes|No|Mixed", "notes": "" }}
  }},
  "goals_training": {{
    "short_term_goals": ["..."],
    "medium_long_term_goals": ["..."],
    "training_needs": ["..."]
  }},
  "manager_assessment": {{
    "overall": "Excellent|Very good|Good|Needs improvement|Mixed",
    "salient_quotes": ["..."],
    "validations_raises_last5y": ["..."]
  }},
  "evidence_last5y": {{
    "2021": {{ "key_facts": ["..."], "numbers": {{}} }},
    "2022": {{ ... }},
    "2023": {{ ... }},
    "2024": {{ ... }},
    "2025": {{ ... }}
  }},
  "decision_features": {{
    "fit_axes": {{
      "teaching_exec_MBA": "high|med|low",
      "research_rank2plus": "high|med|low",
      "leadership_mgmt": "high|med|low",
      "institutional_outreach": "high|med|low"
    }},
    "risk_flags": ["..."],
    "differentiators": ["..."],
    "gaps_vs_strong_offer": ["..."]
  }},
  "trend_notes": "Trajectory over last 5 years",
  "last_updated_year": 20XX,
  "confidence": "high|medium|low"
}}
"""

def build_match_expl_prompt(job_description: Dict,
                            candidate: Dict[str, Any]) -> str:
    """
    Prompt pour obtenir une explication FR/EN alignée sur l'offre.

    Objectif : retourner uniquement
    - summary.skills_* (matched / missing)
    - summary.experience_* (matched / missing)
    - summary.qualifications_* (matched / missing)
    - reasoning
    - suggestions (programmes / certifs / upskilling concrets)
    """
    import json

    candidate_name = candidate.get("name", "Candidate")
    profile_text = candidate.get("profile_text", "")

    return f"""
You are an AI assistant specialized in HR matching and talent assessment.

Respond ONLY with ONE valid JSON object (no markdown, no comments, no extra text).

OVERALL GOAL
Compare this job description and this candidate profile. Your job is to:
- Identify which required skills, experiences and qualifications from the job are PRESENT in the candidate profile (matched).
- Identify which required skills, experiences and qualifications from the job are NOT FOUND in the candidate profile (missing).
- Produce a concise, non-generic reasoning that highlights the key fit and gaps.
- Propose realistic upskilling suggestions (programs, trainings, certifications, projects).

GENERAL CONSTRAINTS
- Do NOT calculate any score. Always set "match_score": null.
- Use ONLY information that appears in the job description or candidate profile. No hallucinations, no guessing.
- Never invent technologies, diplomas, years of experience, company names or responsibilities.
- An item MUST NOT appear in both *_matched and *_missing.
- Avoid HR clichés and long sentences. Prefer short, precise phrases.
- When you say something is matched, it must be grounded in a concrete element from the candidate profile.

MATCHING PROTOCOL (INTERNAL)
1) Extract the required skills, experiences and qualifications from the job.
2) Extract those actually mentioned in the candidate profile.
3) For each job requirement, decide:
   - Matched  → there is clear evidence in the candidate profile.
   - Missing  → there is no evidence in the candidate profile.
4) Fill the JSON fields accordingly.

CATEGORIES & DEFINITIONS
You must classify information into:
- skills_matched / skills_missing
- experience_matched / experience_missing
- qualifications_matched / qualifications_missing

Definitions:
- "skills" = tools, technologies, methods, soft skills mentioned, languages, frameworks, platforms, cloud providers, etc.
- "experience" = missions, responsibilities, domains, contexts (e.g. teaching, industry, SaaS, CI/CD, support), seniority level.
- "qualifications" = diplomas, degrees, certifications, formal trainings.

HOW TO WRITE ITEMS IN SUMMARY ARRAYS
- Each element of the *_matched and *_missing arrays MUST be a very short, targeted phrase (not a long sentence).
- Focus on the essence of the requirement + a hint of context if useful.
- No need to répéter “required by the job” / “mentioned in the profile” à chaque fois.

Examples of STYLE (not mandatory text):
- skills_matched:
  - "Python (APIs, backend)"
  - "React"
  - "Azure, Docker, CI/CD (Azure DevOps)"
- skills_missing:
  - "JavaScript (ES6+)"
  - "TypeScript"
- experience_matched:
  - "SaaS platform for B2B clients"
  - "CI/CD pipeline optimisation, Azure DevOps"
- experience_missing:
  - "Frontend development in JavaScript"
- qualifications_matched:
  - "BSc in Software Engineering"
- qualifications_missing:
  - "Cloud certification (Azure/AWS)"

REACT / JAVASCRIPT AND SIMILAR CASES (IMPORTANT)
- Treat each requirement separately.
- Do NOT assume that one technology implies another.
- Example:
  - Job requires "React" and "JavaScript".
  - Profile mentions "React" but not "JavaScript":
    - Put "React" in skills_matched.
    - Put "JavaScript" in skills_missing.
    - In the reasoning, you briefly note: "React present, JavaScript not stated."

Apply similar logic for:
- TypeScript vs JavaScript
- AWS vs generic cloud
- SQL vs specific DBs (PostgreSQL, MySQL, etc.).

REASONING SECTION (VALUE FOR THE ROLE / COMPANY)
For each language ("en" and "fr"):
- 3 to 4 short paragraphs.
- The reasoning MUST NOT:
  - Repeat the detailed lists of skills, experiences or qualifications.
  - Enumerate all missing items (this is already in the summary).
- The reasoning MUST focus on:
  - Why this profile makes sense (or not) for this specific role and environment.
  - How the candidate’s past contexts (type of companies, products, users, constraints) can translate into value for the team or company.
  - On which types of missions or responsibilities the candidate could be quickly useful in the first months.
  - What kind of mindset, maturity or way of working they are likely to bring (e.g. autonomy, structure, experimentation, pedagogy, support to juniors).
  - 2–3 key points of attention or adaptation to secure a good integration (without re-listing every missing skill).

You MUST keep the tone factual and oriented toward decision-making, as if you were explaining to a hiring manager why this person should or should not be shortlisted.

SUGGESTIONS
- 1 to 3 items in "suggestions" per language.
- Each suggestion MUST correspond to something present in one of the *_missing arrays.
- Focus on concrete, realistic actions: training programs, technologies to learn, certifications, or portfolio/side-project ideas.

Example of STYLE:
- {{ "missing": "JavaScript (frontend)", "recommendation": "Follow a modern JavaScript + React course and build 1–2 frontend projects to add to the CV." }}
- {{ "missing": "Cloud certification", "recommendation": "Prepare an Azure Developer or Azure Administrator certification to formalize existing cloud skills." }}

OUTPUT FORMAT (EXACT KEYS, EXACT SHAPE)
You MUST respect this JSON structure exactly. Only fill the array values and text strings.

{{
  "en": {{
    "name": "{candidate_name}",
    "match_score": null,
    "summary": {{
      "skills_matched": [],
      "skills_missing": [],
      "experience_matched": [],
      "experience_missing": [],
      "qualifications_matched": [],
      "qualifications_missing": []
    }},
    "reasoning": "<3–4 courts paragraphes en français décrivant la pertinence globale du profil pour le poste et l’entreprise : comment son parcours, ses contextes précédents et sa manière de travailler peuvent apporter de la valeur, sur quels types de missions il peut être rapidement utile, quelle dynamique ou maturité il peut apporter à l’équipe, et quels 2–3 points d’adaptation ou de vigilance seraient à prévoir pour sécuriser son intégration.>",
    "suggestions": [
      {{ "missing": "...", "recommendation": "..." }}
    ]
  }},
  "fr": {{
    "name": "{candidate_name}",
    "match_score": null,
    "summary": {{
      "skills_matched": [],
      "skills_missing": [],
      "experience_matched": [],
      "experience_missing": [],
      "qualifications_matched": [],
      "qualifications_missing": []
    }},
    "reasoning": "<3–4 courts paragraphes en français décrivant la pertinence globale du profil pour le poste et l’entreprise : comment son parcours, ses contextes précédents et sa manière de travailler peuvent apporter de la valeur, sur quels types de missions il peut être rapidement utile, quelle dynamique ou maturité il peut apporter à l’équipe, et quels 2–3 points d’adaptation ou de vigilance seraient à prévoir pour sécuriser son intégration.>",
 "suggestions": [
      {{ "missing": "...", "recommendation": "..." }}
    ]
  }}
}}

WRITE THE JSON NOW. Do not include any explanation outside of this JSON.

Job Description (authoritative, JSON):
{json.dumps(job_description, ensure_ascii=False)}

Candidate object (from filtered_ranked, JSON):
{json.dumps(candidate, ensure_ascii=False)}

Candidate profile text (authoritative evidence for the candidate):
{profile_text}
""".strip()

def job_info_prompt(job_text: str) -> str:
    return f"""
You are an elite HR and data extraction assistant. Output a SINGLE valid JSON object.
It MUST contain the top-level keys "en", "fr", and a top-level "react_icon_import".
Do NOT place "react_icon_import" inside "en" or "fr". It MUST appear ONLY once at the top-level.

Schema (top-level keys only):
{{
  "en": {{
    "job_title": {{
      "raw": "string",
      "normalized": "string",
      "seniority": "string|null",
      "specializations": ["string"],
      "aliases": ["string"]
    }},
    "short_description": {{
      "summary": "string",
      "team_context": "string|null",
      "business_impact": "string|null",
      "domain_keywords": ["string"]
    }},
    "responsibilities": [
      {{
        "statement": "string",
        "category": "string",
        "kpis": ["string"],
        "frequency": "string|null",
        "time_share_pct": "number|null",
        "tools": ["string"],
        "seniority_tag": "string|null"
      }}
    ],
    "required_skills": [
      {{
        "name": "string",
        "normalized": "string",
        "type": "string",
        "category": "string",
        "subskills": ["string"],
        "versions_or_flavors": ["string"],
        "proficiency": "string|null",
        "years_min": "number|null",
        "recency_months_max": "number|null",
        "must_have": "boolean",
        "weight": "number"
      }}
    ],
    "desired_experience": {{
      "years_min": "number|null",
      "years_max": "number|null",
      "industries": ["string"],
      "domains": ["string"],
      "methodologies": ["string"],
      "environments": ["string"],
      "leadership": {{
        "people_managed_min": "number|null",
        "projects_led_min": "number|null"
      }},
      "team_size_range": ["number", "number"]|null,
      "client_facing": "boolean|null",
      "travel_percent_max": "number|null",
      "work_model": "string|null",
      "location": {{
        "city": "string|null",
        "region_or_state": "string|null",
        "country": "string|null"
      }},
      "security_clearance": ["string"],
      "portfolio_or_code_samples_required": "boolean|null",
      "notable_project_examples": ["string"]
    }},
    "qualifications": {{
      "education": {{
        "degree_level_min": "string|null",
        "fields_of_study": ["string"]
      }},
      "certifications": [
        {{
          "name": "string",
          "issuer": "string|null",
          "required": "boolean"
        }}
      ],
      "licenses": ["string"],
      "languages": [
        {{
          "name": "string",
          "level": "string|null"
        }}
      ],
      "work_authorization": [
        {{
          "country": "string",
          "required": "boolean"
        }}
      ],
      "background_checks": ["string"],
      "physical_requirements": ["string"],
      "other": ["string"]
    }}
  }},
  "fr": {{ /* same structure as 'en', in French */ }},
  "react_icon_import": "string"
}}

Icon rules (MUST produce exactly one valid import line):
- Format EXACT: "import {{ <IconName> }} from 'react-icons/<pack>'";
- <pack> ∈ {{ ai, fa, md, bs, bi, hi, hi2, io, io5, tb, cg, vsc, sl, ti, wi, ci, lu, rx }}
- Mapping:
  - Developers/Engineering → "import {{ MdEngineering }} from 'react-icons/md';"
  - Data/Analytics/AI → "import {{ AiOutlineAreaChart }} from 'react-icons/ai';"
  - Product/Project/PM → "import {{ MdOutlineDashboard }} from 'react-icons/md';"
  - Design/UX → "import {{ MdDesignServices }} from 'react-icons/md';"
  - HR/People Ops → "import {{ MdPeopleOutline }} from 'react-icons/md';"
  - Finance/Accounting → "import {{ MdAttachMoney }} from 'react-icons/md';"
  - Marketing/Sales → "import {{ MdCampaign }} from 'react-icons/md';"
- If uncertain, ALWAYS use: "import {{ AiOutlineInfoCircle }} from 'react-icons/ai';"

Return ONLY JSON (no markdown, no prose).

Text:
{job_text}
""".strip()

def build_cv_prompt(cv_text: str) -> str:
    """
    Extraction CV orientée matching (FR/EN mix). 
    Objectif: produire un JSON propre, normalisé, pondéré (weights/recency) pour booster les embeddings et la similarité cosine côté matching.
    - Aucune hallucination: si une info est absente → omettre le champ (ne pas mettre de placeholder).
    - Normaliser les intitulés/compétences en anglais côté "normalized".
    - Dates au format ISO: YYYY ou YYYY-MM.
    - Dédupliquer les listes, conserver 3–12 éléments les plus pertinents par section.
    - Sortie: UN SEUL objet JSON (pas de markdown, pas de texte autour).
    """
    return f"""
You are a senior HR resume parser. Read ONE candidate CV (possibly messy, mixed FR/EN).

Output: RETURN A SINGLE VALID JSON OBJECT. DO NOT add commentary or markdown. 
Only include fields you can extract with high confidence. If unknown, omit the field.

NORMALIZATION RULES
- Titles/skills normalization: provide canonical labels in English under "normalized" keys.
- Keep any raw strings as-is under "raw".
- Dates: "YYYY" or "YYYY-MM".
- Lists: deduplicate; keep most relevant 3–12 items.
- Skills: include per-skill recency (last_used) and weight (0–1) based on emphasis in CV (explicit emphasis → higher weight).
- Do not invent employers, titles, dates, or numbers.

SCHEMA (omit any missing fields):

{{
  "identity": {{
    "full_name": "string",
    "current_title": {{
      "raw": "string",
      "normalized": "string"  # e.g., "Data Analyst"
    }},
    "seniority": "Intern|Junior|Mid|Senior|Lead|Manager|Director|VP|Professor|Researcher",
    "specializations": ["string"],               # e.g., "Backend", "Computer Vision"
    "location_current": "City, Country",
    "work_authorization": ["Country"],           # countries where candidate can legally work (if stated)
    "relocation_open": true,                     # only if explicitly stated
    "remote_preference": "onsite|hybrid|remote", # only if stated
    "summary_one_line": "short factual blurb"
  }},

  "contacts": {{
    "email": "string",
    "phone": "string",
    "location": "City, Country",
    "linkedin": "url",
    "github": "url",
    "portfolio": "url"
  }},

  "education": [
    {{
      "degree": "string",
      "field": "string",
      "institution": "string",
      "start": "YYYY or YYYY-MM",
      "end": "YYYY or YYYY-MM",
      "location": "City, Country",
      "honors": ["string"]
    }}
  ],

  "experience": [
    {{
      "title": {{
        "raw": "string",
        "normalized": "string"
      }},
      "company": "string",
      "industry": "string",                      # if stated (e.g., "Fintech")
      "location": "City, Country",
      "employment_type": "full-time|part-time|contract|internship|freelance",  # if stated
      "start": "YYYY or YYYY-MM",
      "end": "YYYY or YYYY-MM|present",
      "team_size": "number",                     # if stated
      "leadership": "IC|Lead|Manager|null",      # if stated
      "achievements": [                          # bullet points with impact/metrics if present
        "string"
      ],
      "tools_tech": ["string"],                  # tech/tools explicitly used
      "methodologies": ["string"],               # e.g., "Agile", "Scrum", "TDD"
      "domains": ["string"],                     # e.g., "e-commerce", "NLP"
      "clients_or_products": ["string"]          # if named in CV
    }}
  ],

  "skills": {{
    "hard": [
      {{
        "name": "string",                        # as in CV (e.g., "Python")
        "normalized": "string",                  # canonical label (e.g., "Python")
        "category": "Programming|Data|ML|Cloud|Web|Mobile|Security|DevOps|DB|QA|Design|PM|Other",
        "subskills": ["string"],                 # e.g., ["NumPy","Pandas"]
        "versions_or_flavors": ["string"],       # e.g., ["React 18","AWS EC2"]
        "proficiency": "basic|intermediate|advanced|expert|null",
        "years_experience": "number|null",       # only if stated or clearly derivable from dates
        "last_used": "YYYY or YYYY-MM|null",     # if stated or implied by most recent role
        "weight": 0.0                            # 0–1 importance inferred from emphasis in CV (prioritize if repeated, in title, or achievements)
      }}
    ],
    "soft": ["string"],                           # e.g., "communication", "leadership" (only if explicitly present)
    "languages": [
      {{"language": "string", "proficiency": "Native|C2|C1|B2|B1|A2|A1"}}
    ]
  }},

  "projects": [
    {{
      "name": "string",
      "role": "string",
      "summary": "string",
      "tech_stack": ["string"],
      "outcomes": ["string"],                    # measurable results if present
      "links": ["url"]
    }}
  ],

  "certifications": [
    {{
      "name": "string",
      "issuer": "string",
      "year": "YYYY"
    }}
  ],

  "awards": ["string"],
  "publications": ["string"],
  "volunteering": ["string"],

  "derived": {{
    "years_experience_total": "number",          # computed from experience dates if possible
    "primary_domains": ["string"],               # top 3–5 domains seen across roles/projects
    "main_technologies": ["string"]              # top 5–10 hard skills/tech across CV
  }},

  "decision_features": {{
    "fit_axes": {{
      "seniority_band": "Junior|Mid|Senior|Lead|Management",
      "domain_expertise": ["string"],            # domains with strongest evidence
      "core_technologies": ["string"],           # tech with highest weights/recency
      "management_level": "IC|Lead|Manager|Director+|null",
      "client_facing": "low|medium|high|null",   # only if clearly stated
      "language_fit": ["string"],                # e.g., ["English:C1","French:B2"]
      "location_fit": "strong|medium|weak|null"  # only if deducible (e.g., same city/country stated)
    }},
    "priority_keywords": ["string"],             # 10–25 high-signal keywords from CV (tech, domains, certifications)
    "dealbreakers": ["string"]                   # ONLY if explicitly stated (e.g., "cannot travel", "needs visa sponsorship")
  }},

  "summary_long": "2–3 short factual paragraphs: who they are, seniority, strongest skills (by weight+recency), notable achievements/industries."
}}

INPUT CV:
{cv_text}
""".strip()

def normalize_skema_url(url: str) -> str:
    """
    Corrige l'URL pour qu'elle pointe vers https://www.skema.edu/en/...
    en gérant le cas où on a seulement un chemin relatif comme
    /executive-education/certificates/...
    """
    if not url:
        return None

    url = url.strip()

    if url.startswith("http://") or url.startswith("https://"):
        if "www.skema.edu" in url and "/en/" not in url.split("?")[0]:
            base, *rest = url.split("www.skema.edu", 1)
            path = rest[0]
            if not path.startswith("/"):
                path = "/" + path
            if not path.startswith("/en/"):
                path = "/en" + path
            return base + "www.skema.edu" + path
        return url

    if not url.startswith("/"):
        url = "/" + url

    if "/en/" not in url.split("?")[0]:
        if not url.startswith("/en/"):
            url = "/en" + url

    return "https://www.skema.edu" + url

def build_skema_certificate_prompt(url: str, html: str) -> str:
    """
    Construit un PROMPT TEXTE pour que le LLM :
    - analyse la page LISTE des certificats SKEMA
    - retourne un JSON avec une liste "certificats": [...]
      (name, url, description, etc.)

     IMPORTANT :
    - Cette fonction doit renvoyer une STRING (et pas une liste de messages),
      car _llm(prompt: str) envoie déjà messages=[{"role": "user", "content": prompt}].
    """
    listing_url = normalize_skema_url(url) or url

    return f"""
You are an assistant that extracts structured data about SKEMA Business School
executive education certificates from the HTML of a LISTING page.

You MUST return ONLY a valid JSON object. No explanations, no markdown.

The JSON MUST have the following top-level structure:

{{
  "certificats": [
    {{
      "name": <string>,                 // certificate title
      "url": <string>,                  // FULL https URL (starting with "https://www.skema.edu")
      "description": <string or null>,  // short description if available, else null
      "online_or_location": <string or null>, // e.g. "Online", "Paris", "Online or on site", else null
      "domain": <string or null>,       // e.g. "Project Management", "Finance", etc., else null
      "source": "skema"
    }},
    ...
  ]
}}

CRITICAL INSTRUCTIONS – DO NOT IGNORE:

1. You receive the HTML of ONE LISTING page that contains MULTIPLE certificates.

2. You MUST scan the ENTIRE HTML and identify **EVERY** certificate on the page.
   A "certificate" is ANY item whose detail page URL path contains
   "executive-education/certificates/" (absolute or relative), for example:
     - "/en/executive-education/certificates/..."
     - "/executive-education/certificates/..."
     - "https://www.skema.edu/en/executive-education/certificates/..."

   For EVERY unique such URL, you MUST create ONE entry in "certificats"
   (even if there is very little text around it).

3. For EACH certificate you identify, extract:
   - its name/title (use the visible text of the link or card; if several texts,
     choose the most specific title-like text)
   - its detail page URL
   - a short description if any paragraph or text block is clearly associated to it
   - information about delivery mode or location if any
   - a domain or main thematic area if any

4. If you cannot find a specific field for a given certificate:
   - set its value to null (NOT an empty string), except "source" which must always be "skema".
   - DO NOT skip the certificate just because description / location / domain are missing.
     As long as you have at least a title + URL, you MUST include it.

5. URLs:
   - If the HTML contains RELATIVE URLs (like "/en/executive-education/certificates/..."),
     you MUST convert them into ABSOLUTE URLs starting with "https://www.skema.edu".
   - If the HTML already contains ABSOLUTE URLs to SKEMA, keep them but ensure they start with
     "https://www.skema.edu" and, if needed, insert "/en/" after the domain so that the path
     is of the form "/en/...".
   - Normalize all URLs using this rule so that each certificate has exactly one normalized URL.

6. Deduplication:
   - If the SAME certificate appears several times on the page (same normalized URL),
     you MUST keep only ONE object in the "certificats" array for that URL.
   - When merging duplicates, keep the richest information you can infer for each field.

7. DO NOT include items that are clearly not executive education certificates
   (for example: footer navigation, generic links, news, blog posts, etc.).

Listing page canonical URL:
{listing_url}

Return STRICTLY a JSON object with a "certificats" array as described above.

Here is the HTML content of the LISTING page:

<<<HTML>
{html}
</HTML>>>
""".strip()

def _build_details_prompt(detail_url: str, html: str) -> str:
    """
    Prompt LLM pour UNA seule formation (page détail).
    Le modèle doit extraire / corriger:
      - target_audience
      - duration
      - date
      - fees
      - contact

    Si une info est introuvable dans le HTML, il met null.
    """
    return f"""
You are an assistant that extracts structured data about ONE SKEMA certificate
from the HTML of its dedicated page.

You MUST return ONLY a valid JSON object. No explanation, no markdown.

Canonical URL of this certificate:
{detail_url}

Using ONLY the information present in the HTML below, extract the following fields:

- "target_audience": main target audience of the certificate
    (example: "Project managers, product owners, team leaders") or null
- "duration": duration of the programme
    (example: "3 days", "2 x 3 days", "6 weeks online") or null
- "date": main upcoming session dates or typical period
    (example: "From 10 to 12 June 2025", "Several intakes per year") or null
- "fees": tuition fees or price
    (example: "€2,900 excl. VAT") or null
- "contact": contact email, phone number or a short summary of the contact section
    (example: "executive-education@skema.edu", "+33 (0)1...") or null

If you cannot find a field clearly in the HTML, set its value to null (NOT an empty string).

Return STRICTLY a JSON object with this schema:

{{
  "url": "{detail_url}",
  "target_audience": <string or null>,
  "duration": <string or null>,
  "date": <string or null>,
  "fees": <string or null>,
  "contact": <string or null>
}}

Here is the HTML content of the page:

<<<HTML>
{html}
</HTML>>>
""".strip()



