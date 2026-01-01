# Handbook QA System – Mini-Capstone (DSC 360)

## Overview
This project implements a retrieval-augmented question-answering system using 
selected excerpts from the Centre College Student Handbook. The system retrieves 
the most relevant policy chunk and generates a controlled, safe answer using 
prompt-based guardrails.

## How to Run the Application
1. Activate the appropriate Python environment on the lab machine.
2. Navigate to the project folder:
   cd project/
3. Run:
   python3 src/ask.py
4. Type any question to see the system response.

## How to Run the Evaluation
The evaluation script uses gold.csv to produce results.csv:
   python3 src/score.py

## Folder Structure
project/
│
├── data/            # raw text handbook files
│   ├── *.txt
│
├── index/           # built index artifacts
│   ├── chunks.jsonl
│   ├── embeddings.npy
│   ├── meta.json
│   └── questions.jsonl
│
├── out/             # evaluation output
│   └── results.csv
│
├── src/             # Python source files
│   ├── ask.py
│   ├── index.py
│   ├── ingest.py
│   └── score.py
│
├── tests/           # evaluation file
│   └── gold.csv
│
└── README.md

