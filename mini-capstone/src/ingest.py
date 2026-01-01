"""
ingest.py

Reads all .txt files in ../data, splits them into chunks, and writes
../index/chunks.jsonl with records of the form:

{
    "chunk_id": "chunk_0",
    "section_title": "Sexual Misconduct Title Ix",
    "source_file": "sexual_misconduct_title_ix.txt",
    "text": "..."
}

Chunking logic:
- Collapses runs of blank lines (2, 3, 4 Enters) into a single blank line.
- Splits on blank lines into paragraphs.
- Groups paragraphs into chunks up to ~MAX_CHARS characters.
"""

import json
import os
from pathlib import Path
import re


# Target size per chunk in characters (roughly 500–700 words)
MAX_CHARS = 3000

# filenames that contain multiple sections with Roman numeral headings
MULTI_SECTION_FILES = {
    "student_conduct_code.txt",
    "sexual_misconduct_title_ix.txt",
    "activities_and_organizations.txt",  
    "living_on_campus.txt",
}

# handbook URLs per file
URLS = {
    "activities_and_organizations.txt": "https://centre.smartcatalogiq.com/en/catalog-and-handbooks/student-handbook/activities-and-organizations",
    "alcohol_drug_policy.txt": "https://centre.smartcatalogiq.com/catalog-and-handbooks/student-handbook/student-policies-and-regulations/alcohol-and-drug-policy",
    "good_samaritan_medical_amnesty.txt": "https://centre.smartcatalogiq.com/en/catalog-and-handbooks/student-handbook/student-policies-and-regulations/good-samaritan-and-medical-amnesty-policies",
    "greek_life_social_policy.txt": "https://centre.smartcatalogiq.com/en/catalog-and-handbooks/student-handbook/student-policies-and-regulations/greek-life-social-policy",
    "hazing_statement.txt": "https://centre.smartcatalogiq.com/en/catalog-and-handbooks/student-handbook/student-policies-and-regulations/hazing-statement",
    "living_on_campus.txt": "https://centre.smartcatalogiq.com/en/catalog-and-handbooks/student-handbook/living-on-campus",
    "student_conduct_code.txt": "https://centre.smartcatalogiq.com/en/catalog-and-handbooks/student-handbook/student-policies-and-regulations/policy-prohibiting-discrimination-and-harassment",
    "sexual_misconduct_title_ix.txt": "https://centre.smartcatalogiq.com/en/catalog-and-handbooks/student-handbook/student-policies-and-regulations/sexual-misconduct-policy",
    "student_travel_policy.txt": "https://centre.smartcatalogiq.com/en/catalog-and-handbooks/student-handbook/student-policies-and-regulations/student-travel-policy",
    "visitation_policy.txt": "https://centre.smartcatalogiq.com/en/catalog-and-handbooks/student-handbook/student-policies-and-regulations/visitation",
}

ROMAN_HEADER_RE = re.compile(r"^(?P<num>[IVXLCDM]+)\.\s+(?P<title>.+)$")



def load_text_files(data_dir: Path):
    """
    Yield (filename, text) for every .txt file in data_dir.
    """
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        yield path.name, text



def normalize_newlines(text: str) -> str:
    """
    Collapse sequences of 2+ blank lines down to exactly one blank line.

    It doesn't matter if the original handbook or you used
    2, 3, or 5 Enters between sections; they all become a single blank
    line, which we treat as a paragraph break.
    """
    lines = text.splitlines()
    normalized_lines = []
    blank_streak = 0
    for line in lines:
        if line.strip() == "":
            blank_streak += 1
            # only keep a single blank line in a streak
            if blank_streak == 1:
                normalized_lines.append("")
        else:
            blank_streak = 0
            normalized_lines.append(line)

    return "\n".join(normalized_lines)



def _split_paragraphs_into_chunks(paragraphs, max_chars: int):
    current = []
    cur_len = 0

    for p in paragraphs:
        p_len = len(p)
        if current and cur_len + p_len + 2 > max_chars:
            yield "\n\n".join(current)
            current = [p]
            cur_len = p_len
        else:
            current.append(p)
            cur_len += p_len + 2

    if current:
        yield "\n\n".join(current)



def iter_section_chunks(text: str, filename: str, max_chars: int = MAX_CHARS):
    """
    Yield (section_title, chunk_text) pairs.

    For multi-section files:
      - detect Roman numeral headers line-by-line, e.g. "I. Title"
      - use the header line's title as section_title
      - exclude the header line itself from the section body

    For other files:
      - use filename-based title for all chunks.
    """
    text = normalize_newlines(text)
    multi = filename in MULTI_SECTION_FILES
    base_title = os.path.splitext(filename)[0].replace("_", " ").title()

    if not multi:
        # simple: whole file under one title
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for chunk_text in _split_paragraphs_into_chunks(paragraphs, max_chars):
            yield base_title, chunk_text
        return

    lines = text.splitlines()
    current_title = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        m = ROMAN_HEADER_RE.match(stripped)
        if m:
            # flush any previous section
            if current_title and current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    paragraphs = [
                        p.strip()
                        for p in section_text.split("\n\n")
                        if p.strip()
                    ]
                    for chunk_text in _split_paragraphs_into_chunks(paragraphs, max_chars):
                        yield current_title, chunk_text

            # start new section; do NOT include the header line in content
            current_title = m.group("title").strip()
            current_lines = []
        else:
            # normal content line
            current_lines.append(line)

    # flush last section
    if current_lines:
        title = current_title if current_title else base_title
        section_text = "\n".join(current_lines).strip()
        if section_text:
            paragraphs = [
                p.strip()
                for p in section_text.split("\n\n")
                if p.strip()
            ]
            for chunk_text in _split_paragraphs_into_chunks(paragraphs, max_chars):
                yield title, chunk_text




def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    index_dir = root / "index"
    index_dir.mkdir(exist_ok=True)

    out_path = index_dir / "chunks.jsonl"

    chunk_id = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for filename, text in load_text_files(data_dir):
            url = URLS.get(filename, "")
            for section_title, chunk_text in iter_section_chunks(text, filename, MAX_CHARS):
                record = {
                    "chunk_id": f"chunk_{chunk_id}",
                    "section_title": section_title,
                    "source_file": filename,
                    "url": url,
                    "text": chunk_text,
                }
                f_out.write(json.dumps(record) + "\n")
                chunk_id += 1

    print(f"Wrote {chunk_id} chunks to {out_path}")


if __name__ == "__main__":
    main()