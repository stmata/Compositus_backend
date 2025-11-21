def extract_candidate_profile_text(profile: dict) -> str:
    """
    Converts structured candidate data into a freeform string for LLM input.
    """
    sections = []

    def year_range(data: dict):
        years = [int(y) for y in data if data[y]]
        return f"{min(years)}–{max(years)}" if years else "No active years"

    job_titles = profile.get("Job Title", {})
    job_summary = ", ".join(
        f"{title} ({y})" for y, title in job_titles.items() if title and title != "nan"
    )
    sections.append(f"Job Titles and Years: {job_summary}")

    for section in [
        "Skills", "Evaluation", "Pedagogical Performance",
        "Supervision Performance", "Intellectuel Performance",
        "Career Goal Short Term", "Career Goal Mid/Long Term",
        "Academic/Admin Responsibilities", "Research Areas",
        "Experience", "Projects"
    ]:
        yearly_data = profile.get(section, {})
        entries = [f"{y}: {text}" for y, text in yearly_data.items() if text and text != "nan"]
        if entries:
            sections.append(f"{section}:\n" + "\n".join(entries))

    return "\n\n".join(sections)
