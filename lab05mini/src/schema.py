# src/schema.py
"""
Part 1 — Schema (simple & strict where it matters)


Mini-report (what this enforces):
- program: exactly 3 uppercase letters (e.g., CSC, ECO, THR)
- number: 3 digits in 100–599 range with optional L suffix (e.g., 210, 210L)
- section: one lowercase letter or blank
- title: non-empty string (we keep the source casing)
- credits: float 0.0–6.0; accepts ranges like "1-3" upstream but stores the FIRST number (e.g., 1.0)
- days: exactly 7 chars in UMTWRFS order (e.g., -M-W-F-); blank allowed
- times: "H:MM-H:MMAM/PM" (AM/PM can appear on either/both ends); blank allowed
- room: campus pattern "AAAA 123" or blank (TBA/------- → blank)
- faculty: non-empty string
- tags: comma-separated codes that may contain letters/digits/*; we uppercase
"""




from __future__ import annotations
from pydantic import BaseModel, field_validator
from typing import Optional, Any
import re


# Treat these as missing
_TBA_STRINGS = {"TBA", "tba", "-------", "", None}


class SectionRow(BaseModel):
   program: str               # e.g., "CSC"
   number: str                # e.g., "210" or "210L"
   section: Optional[str]     # single lowercase letter or None
   title: str                 # free text (kept as-is)
   credits: float             # 0.0–6.0
   days: Optional[str]        # 7-char mask like -M-W-F- in UMTWRFS order
   times: Optional[str]       # e.g., 1:50-2:50PM
   room: Optional[str]        # e.g., YOUN 101
   faculty: str               # non-empty
   tags: Optional[str] = None # e.g., "E1", "BLAP", "A**" or comma-list like E1,A


   # ----- normalize nullable text fields -----
   @field_validator("section", "days", "times", "room", "tags", mode="before")
   @classmethod
   def _opt(cls, v: Any) -> Optional[str]:
       if v is None:
           return None
       if isinstance(v, str):
           s = v.strip()
           return None if s in _TBA_STRINGS else (s if s else None)
       return v


   # ---------- program ----------
   @field_validator("program")
   @classmethod
   def _program(cls, v: str) -> str:
       s = v.strip()
       if not re.fullmatch(r"[A-Z]{3}", s):
           raise ValueError("program must be exactly three uppercase letters (e.g., CSC).")
       return s


   # ---------- number ----------
   @field_validator("number")
   @classmethod
   def _number(cls, v: str) -> str:
       s = v.strip().upper()
       if not re.fullmatch(r"[1-5]\d{2}L?", s):
           raise ValueError("number must be 100–599 with optional 'L' (e.g., 210 or 210L).")
       return s


   # ---------- section ----------
   @field_validator("section")
   @classmethod
   def _section(cls, v: Optional[str]) -> Optional[str]:
       if v is None:
           return None
       if not re.fullmatch(r"[a-z]", v):
           raise ValueError("section must be one lowercase letter (e.g., a).")
       return v


   # ---------- title ----------
   @field_validator("title", mode="before")
   @classmethod
   def _title(cls, v: Any) -> str:
       if not isinstance(v, str):
           raise ValueError("title must be a string.")
       s = v.strip()
       if not s:
           raise ValueError("title cannot be empty.")
       return s


   # ---------- credits ----------
   @field_validator("credits")
   @classmethod
   def _credits(cls, v: Any) -> float:
       # Accept "1", "3.0", or ranges like "1-3" (take the first number)
       if isinstance(v, (int, float)):
           c = float(v)
       else:
           s = str(v).strip()
           m = re.fullmatch(r"([0-6](?:\.\d+)?)(?:-[0-6](?:\.\d+)?)?", s)
           if not m:
               raise ValueError("credits must be numeric or a range like '1-3'.")
           c = float(m.group(1))
       if not (0.0 <= c <= 6.0):
           raise ValueError("credits must be between 0.0 and 6.0.")
       return c


   # ---------- days ----------
   @field_validator("days")
   @classmethod
   def _days(cls, v: Optional[str]) -> Optional[str]:
       if v is None:
           return None
       s = v.strip().upper()
       # Correct order per your note: U M T W R F S
       if not re.fullmatch(r"[UMTWRFS-]{7}", s):
           raise ValueError("days must be a 7-char mask like -M-W-F-.")
       return s


   # ---------- times ----------
   @field_validator("times")
   @classmethod
   def _times(cls, v: Optional[str]) -> Optional[str]:
       if v is None:
           return None
       pat = r"^\d{1,2}:\d{2}(?:AM|PM)?-\d{1,2}:\d{2}(?:AM|PM)$"
       if not re.fullmatch(pat, v):
           raise ValueError("times must match H:MM-H:MMAM/PM (e.g., 1:50-2:50PM).")
       return v


   # ---------- room ----------
   @field_validator("room")
   @classmethod
   def _room(cls, v: Optional[str]) -> Optional[str]:
       if v is None:
           return None
       s = v.strip().upper()
       if s in {"TBA", "-------", ""}:
           return None
       if not re.fullmatch(r"[A-Z]{4}\s\d{3}", s):
           raise ValueError("room must look like 'YOUN 101', 'OLIN 208', 'CRNS 134'.")
       return s


   # ---------- faculty ----------
   @field_validator("faculty", mode="before")
   @classmethod
   def _faculty(cls, v: Any) -> str:
       if not isinstance(v, str):
           raise ValueError("faculty must be a string.")
       s = v.strip()
       if not s:
           raise ValueError("faculty cannot be empty.")
       return s


   # ---------- tags ----------
   @field_validator("tags")
   @classmethod
   def _tags(cls, v: Optional[str]) -> Optional[str]:
       if v is None:
           return None
       # Accept tokens like A, E1, BLAP, A**, R, S, comma-separated (commas may have spaces)
       # Normalize: remove spaces after commas, uppercase
       s = v.upper().replace(", ", ",")
       # tokens are letters/digits/* up to 6 chars
       if not re.fullmatch(r"[A-Z0-9\*]{1,6}(?:,[A-Z0-9\*]{1,6})*", s):
           raise ValueError("tags must be comma-separated codes (e.g., E1,BLAP,A**).")
       return s