"""
Strip Project Gutenberg boilerplate from the novels in books/.

The raw files carry a header before the text and the full Gutenberg license
after it. The license alone is ~3% of the corpus, and the RNN learned to recite
it ("This eBook is for the use of anyone anywhere..."). This script writes
cleaned copies to books_clean/ and leaves books/ untouched, because the
existing models' 123-character vocabulary is derived from books/.

What is removed from each book:

* everything up to and including the ``*** START OF THE PROJECT GUTENBERG`` line
* everything from the ``*** END OF THE PROJECT GUTENBERG`` line onward
* the "Produced by ..." credit paragraph
* "End of Project Gutenberg's ..." lines
* ``[Illustration]`` placeholders (the images are not in the text)

Editorial notes such as "[Redactor's Note: ...]" are kept and listed in the
report, since whether they belong in a Verne corpus is a judgment call.

Usage:
    python clean_corpus.py                  # books/ -> books_clean/
    python clean_corpus.py --check          # report only, write nothing
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG", re.M)
END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.M)
# A paragraph (up to the next blank line) that starts with one of these.
DROP_PARAGRAPH = re.compile(r"^(Produced by|\[Illustration)", re.I)
DROP_LINE = re.compile(r"^End of (the )?Project Gutenberg", re.I)
NOTE = re.compile(r"^\[(Redactor|Transcriber|Editor)", re.I)


def clean(raw: str, name: str) -> tuple[str, list[str]]:
    start, end = START.search(raw), END.search(raw)
    if not start or not end:
        raise ValueError(f"{name}: Gutenberg START/END markers not found")
    body = raw[raw.index("\n", start.start()) + 1: end.start()]

    paragraphs = re.split(r"\n\s*\n", body)
    kept, notes = [], []
    for p in paragraphs:
        text = p.strip("\n")
        if not text.strip() or DROP_PARAGRAPH.match(text.strip()):
            continue
        lines = [ln for ln in text.split("\n") if not DROP_LINE.match(ln.strip())]
        if not any(ln.strip() for ln in lines):
            continue
        if NOTE.match(text.strip()):
            notes.append(text.strip().split("\n")[0][:70])
        kept.append("\n".join(lines))
    # Same shape as Gutenberg prose: blank line between paragraphs.
    return "\n\n".join(kept) + "\n", notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=ROOT / "books")
    ap.add_argument("--dst", type=Path, default=ROOT / "books_clean")
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    files = sorted(args.src.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files in {args.src}")
    if not args.check:
        args.dst.mkdir(exist_ok=True)

    raw_all, clean_all = "", ""
    print(f"{'book':<42}{'raw':>10}{'clean':>10}{'removed':>9}")
    for f in files:
        raw = f.read_text(encoding="utf-8")
        cleaned, notes = clean(raw, f.name)
        raw_all += raw + "\n"
        clean_all += cleaned + "\n"
        pct = 100 * (1 - len(cleaned) / len(raw))
        print(f"{f.stem[:41]:<42}{len(raw):>10,}{len(cleaned):>10,}{pct:>8.1f}%")
        for n in notes:
            print(f"    kept note: {n}")
        if not args.check:
            (args.dst / f.name).write_text(cleaned, encoding="utf-8")

    pct = 100 * (1 - len(clean_all) / len(raw_all))
    print(f"{'TOTAL':<42}{len(raw_all):>10,}{len(clean_all):>10,}{pct:>8.1f}%")

    leftover = len(re.findall(r"gutenberg", clean_all, re.I))
    print(f"\n'Gutenberg' mentions left: {leftover}")
    lost = sorted(set(raw_all) - set(clean_all))
    print(f"vocabulary: {len(set(raw_all))} raw, {len(set(clean_all))} clean")
    if lost:
        print(f"characters that only occurred in boilerplate: {lost!r}")
        print("Keep building the vocabulary from books/ so the tokenizer stays at "
              f"{len(set(raw_all))} characters and matches the existing models.")
    if not args.check:
        print(f"\nwrote {len(files)} files to {args.dst.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
