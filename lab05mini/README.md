# Lab 05 Mini — Extracting Structured Data from Unstructured Text

In this lab, you will use a local language model (via **Ollama**) and **Pydantic** validation to turn unstructured text into structured data.

---

## Overview

You are given a plain-text file of course listings such as:

```
CSC 170 Programming and Problem Solving 4 a T. Allen 1:50-3:50PM -M-W-F- OLIN 208
```

Your goal is to build a small extraction app that reads each line, uses a local model to return structured JSON following a schema, validates the result with Pydantic, and writes the output to a CSV file.

---

## Learning Goals

By completing this lab, you will learn to:

* Use Ollama to call a local language model and request structured output.
* Supply a JSON schema to guide model responses.
* Validate model output using Pydantic.
* Write structured data to a CSV file and evaluate accuracy automatically.

---

## Part 1 — Building and Refining the Extractor

1. Open **`src/schema.py`**

   * Add all the fields described in the table below.
   * Implement appropriate data types and validators.
   * Use `None` for optional or missing values (for example, `TBA → None` or `------- → None`).

2. Open **`src/extract.py`**

   * In `extract_structured_record()`, write a prompt for your chosen Ollama model (for example, `gemma3:4b` or `granite3:2b`).
   * Pass `SectionRow.model_json_schema()` as the `format=` argument so the model returns JSON that matches your schema.
   * Parse the model output, validate it with `SectionRow(**data)`, and return the validated object.

3. Run the extractor on the training set:

   ```bash
   cd lab05mini
   python src/extract.py
   ```

   Then open **`out/sections_train.csv`** and inspect your results.
   If your output looks incorrect or incomplete, revise your prompt or schema and run it again.

---

## Schema Guidelines

| Field   | Type  | Nullable | Description                                               |
| ------- | ----- | -------- | --------------------------------------------------------- |
| program | str   | no       | Three-letter uppercase code (e.g. CSC, MAT).              |
| number  | str   | no       | Course number (e.g. “210” or “210L”).                     |
| section | str   | yes      | Single lowercase letter (e.g. a, b).                      |
| title   | str   | no       | Course title in title case.                               |
| credits | float | no       | Numeric values such as 3.0, 4.0, or 0.0.                  |
| days    | str   | yes      | Use `None` if “-------”.                                  |
| times   | str   | yes      | Use `None` if “TBA”.                                      |
| room    | str   | yes      | Building and room (e.g. “OLIN 208”). Use `None` if “TBA”. |
| faculty | str   | no       | Instructor name.                                          |
| tags    | str   | yes      | Optional classification codes such as E1 or E1,A.         |

---

## Part 2 — Testing and Scoring

1. Once you are confident your extractor is producing accurate results, open **`src/extract.py`** and **uncomment** the following line near the bottom:

   ```python
   process_file("raw/testing.txt", "out/sections_test.csv")
   ```

2. Run the extractor again to generate test output:

   ```bash
   python src/extract.py
   ```

3. Evaluate your model with:

   ```bash
   python src/score.py
   ```

   The scoring script compares your predictions against `tests/gold.csv` using **Exact Match (EM)** accuracy.

   * **Field-level EM** measures the proportion of individual fields (columns) that match exactly.
   * **Row-level EM** measures the proportion of complete rows that match in every field.

---

## Checklist

* [ ] Schema defined and validated in `schema.py`.
* [ ] Extraction logic implemented in `extract_structured_record()`.
* [ ] Training output inspected and refined.
* [ ] Test output generated and scored successfully.

---

## Notes

* Requires **Pydantic 2.0 or later**.
  Check your version:

  ```bash
  python -c "import pydantic; print(pydantic.__version__)"
  ```

  If it is below 2.0, update with:

  ```bash
  pip install -U pydantic
  ```

* If you see messages such as “Validation test failed — skipping this line,”
  it means your schema is catching invalid or missing data. This is normal and indicates that your validation is working.

* The provided placeholder extractor produces empty but valid records.
  It allows you to test the pipeline but will score **0.00** until you implement real extraction logic.

