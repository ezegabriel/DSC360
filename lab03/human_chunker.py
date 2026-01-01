import re, os, sys
from pathlib import Path
import typing


def read_book(filename: str) -> str:
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
    
def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    lengths = [len(s) for s in sentences]
    
    longest_sentence = ""
    for sentence in sentences:
        if len(sentence) > len(longest_sentence):
            longest_sentence = sentence
    # print(f"Longest sentence: {longest_sentence}")

    # print(sentences)
    print(f"Sentences: {len(sentences)}")
    print(f"Lens: {lengths}")
    print(f"Min/Mean/Max length (chars): {min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}")

    return [s.strip() for s in sentences if s.strip()]

def strip_gutenberg_boilerplate(text: str) -> str:

    # Remove Project Gutenberg header/footer and typical front-matter noise
    start = re.search(r'\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK', text, re.I)
    end   = re.search(r'\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK', text, re.I)

    if start and end:
        text = text[start.end():end.start()]
    # Drop all-caps one-liners like CONTENTS, CHAPTER headings if they’re alone
    # (we’ll reattach short lines to following paragraphs later)
    return text.strip()

def split_into_paragraphs(text: str):
    paragraphs = text.split("\n\n")

    lengths = [len(p) for p in paragraphs]

    print(f"Paragraphs: {len(paragraphs)}")
    print(f"Lens: {lengths}")
    print(f"Min/Mean/Max length (chars): {min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}")


    return paragraphs


def main():
    input_path = Path("data/book.txt")
    book = read_book(input_path)
    book = strip_gutenberg_boilerplate(book)
    paragraphs = split_into_paragraphs(book)

    # sentences = split_into_sentences(book)
    


if __name__ == "__main__":
    main()