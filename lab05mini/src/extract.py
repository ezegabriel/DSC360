# src/extract.py
from __future__ import annotations


import csv
import re
from typing import List, Optional


from pydantic import ValidationError
from schema import SectionRow




# ------------------------
# TUNABLES
# ------------------------
LIMIT = -1




# ------------------------
# REGEX SNIPPETS (kept hooked on schema.py)
# ------------------------
PROG_NUM_RX = re.compile(r"^(?P<program>[A-Z]{3})\s+(?P<number>[1-5]\d{2}L?)\b")
CREDITS_RX  = re.compile(r"\b(?P<credits>[0-6](?:\.\d+)?(?:-[0-6](?:\.\d+)?)?)\b")
SECTION_RX  = re.compile(r"^\s*(?P<section>[a-z])\b")
TIME_RX     = re.compile(r"\b(?P<times>\d{1,2}:\d{2}(?:AM|PM)?-\d{1,2}:\d{2}(?:AM|PM))\b", re.I)
DAYS_MASK_RX= re.compile(r"\b(?P<days>[UMTWRFS-]{7})\b")  # order U M T W R F S
ROOM_RX     = re.compile(r"\b(?P<room>[A-Z]{4}\s\d{3})\b")




# Standalone day letters (avoid initials like "R." by requiring no trailing dot)
DAY_LETTERS_RX = re.compile(r"(?<![A-Z]\.)\b([UMTWRFS])\b(?!\.)")




# Tags only at very end; tokens must start with a letter (blocks '101' etc.)
TAGS_END_RX = re.compile(
   r"(?:\s+(?P<tags>(?:[A-Za-z][A-Za-z0-9\*]{0,5})(?:\s*,\s*[A-Za-z][A-Za-z0-9\*]{0,5})*))\s*$"
)








def _strip(s: Optional[str]) -> Optional[str]:
   return s.strip() if isinstance(s, str) and s.strip() else None


def parse_line(line: str) -> Optional[SectionRow]:
   """Parse one raw line into a SectionRow. Return None if the line is malformed."""
   src = line.strip()
   if not src:
       return None


   # 1) Program + number at start
   m = PROG_NUM_RX.match(src)
   if not m:
       return None
   program = m.group("program")
   number  = m.group("number")


   rest = src[m.end():].lstrip()


   # 2) Title runs until we hit the credits token
   m = CREDITS_RX.search(rest)
   if not m:
       return None
   title = rest[:m.start()].strip()
   credits_raw = m.group("credits")
   rest2 = rest[m.end():].lstrip()


   # Normalize credits: if "1-3" -> 1.0 (first number)
   credits_first = float(credits_raw.split("-")[0])


   # 3) Optional section (single lowercase letter) right after credits
   section = None
   m = SECTION_RX.match(rest2)
   if m:
       section = m.group("section")
       rest3 = rest2[m.end():].lstrip()
   else:
       rest3 = rest2


   # 4) Faculty runs until we hit either a time range (preferred) or TBA or a days mask
   faculty = None
   times = None
   days = None
   room = None
   tags = None


  


   # Find days/time/room in the remaining string
   # a) time
   tm2 = TIME_RX.search(rest3)
   if tm2:
       times = tm2.group("times")
   # b) days mask
   dm = DAYS_MASK_RX.search(rest3)
   if dm:
       days = dm.group("days")
   # c) room
   rm = ROOM_RX.search(rest3)
   if rm:
       room = rm.group("room")


   # -------- TAGS (only if at very end; don't steal room numbers) --------
   tm = TAGS_END_RX.search(rest3)
   tags = None
   if tm:
       cand = tm.group("tags").strip()
       # If it's only digits (or exactly the digits from a detected room), ignore it
       if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", cand):
           if not (rm and re.fullmatch(r"\d{3}", cand) and cand == rm.group("room").split()[-1]):
               tags = cand
               # remove the tags substring from the end to keep faculty clean if needed
               rest3 = rest3[:tm.start()].rstrip()








   # Faculty is whatever is before the earliest of time/days/room/TBA markers
   # Identify earliest index among these markers
   indices = []
   for mo in [tm2, dm, rm]:
       if mo:
           indices.append(mo.start())
   # Also handle literal TBA tokens that can appear in place of time/days/room
   tba_pos = rest3.find("TBA")
   if tba_pos != -1:
       indices.append(tba_pos)


   if indices:
       cut = min(indices)
       faculty = rest3[:cut].strip()
   else:
       # If nothing else, treat entire rest as faculty (undecided classes might have only faculty)
       faculty = rest3.strip()


   # If days mask missing, try to derive from standalone day letters (UMTWRFS order)
   if not days:
       letters = {d.upper() for d in DAY_LETTERS_RX.findall(rest3)}
       if letters:
           order = list("UMTWRFS")
           days = "".join([c if c in letters else "-" for c in order])


   # Normalize tags: uppercase, remove spaces after commas
   if tags:
       tags = tags.upper().replace(", ", ",").replace(" ,", ",")
       # compress multiple consecutive commas if any weird space pattern
       tags = ",".join([t for t in (x.strip() for x in tags.split(",")) if t])


   # Normalize TBA/------- to None
   times = None if (times is None or times.upper() == "TBA") else times
   days  = None if (days is None or days.upper() == "TBA" or days == "-------") else days
   room  = None if (room is None) else room
   faculty = _strip(faculty)


   # Build and validate
   try:
       return SectionRow(
           program=program,
           number=number,
           section=section,
           title=title,
           credits=credits_first,
           days=days,
           times=times,
           room=room,
           faculty=faculty if faculty else "TBA",  # schema requires non-empty; if unknown, keep "TBA" which schema will accept via normalizer
           tags=tags,
       )
   except ValidationError:
       # Try softer faculty fallback if blank
       return None


def process_file(in_path: str, out_path: str):
   with open(in_path, encoding="utf-8") as fin, open(out_path, "w", newline="", encoding="utf-8") as fout:
       writer = csv.writer(fout, delimiter=";")


       fields: List[str] = list(SectionRow.model_fields.keys())
       writer.writerow(fields)


       n = 0
       for raw in fin:
           if LIMIT >= 0 and n >= LIMIT:
               break
           line = raw.rstrip("\n")
           if not line.strip():
               continue


           rec = parse_line(line)
           if rec is None:
               continue


           row = rec.model_dump()
           writer.writerow([row[k] if row[k] is not None else "" for k in fields])
           n += 1






       print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
   # Training pass
   process_file("raw/training.txt", "out/sections_train.csv")
   # For test later:
   # process_file("raw/testing.txt", "out/sections_test.csv")