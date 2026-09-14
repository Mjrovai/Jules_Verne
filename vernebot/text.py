"""
Post-processing for generated text.

The training corpus is Project Gutenberg's plain-text editions, which are
*hard-wrapped*: prose lines break at about 67 characters, and a blank line
separates paragraphs. The model learned that faithfully, so its raw output
reproduces the wrapping mid-sentence::

    "Tell mind them. The portion of this story against
    December that
    William Guy?" asked my uncle, who was about to overlook
    the end of a
    nervous tone.

For display the soft wraps should become spaces again, while the blank lines
that mark real paragraph breaks must survive. :func:`reflow` does exactly
that, and nothing else — it never changes wording, and it is a pure function,
so it is easy to test and to switch off when the wrapping itself is the lesson.
"""

from __future__ import annotations

__all__ = ["reflow"]


def reflow(text: str) -> str:
    """Turn hard-wrapped prose back into flowing paragraphs.

    A single newline is treated as a soft wrap and replaced by a space; a run of
    two or more newlines is a paragraph break and is preserved. Leading and
    trailing whitespace of each line is stripped, and runs of spaces are
    collapsed to one.
    """
    if not text:
        return text

    # Normalise line endings so the blank-line logic below sees plain "\n".
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    # Normalise runs of spaces per line. The model emits plain spaces, but this
    # keeps the function honest if it is ever fed text with tabs.
    lines = [" ".join(line.split()) for line in normalised.split("\n")]

    out: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            out.append(" ".join(buffer))
            buffer.clear()

    index = 0
    while index < len(lines):
        if lines[index]:
            buffer.append(lines[index])
            index += 1
            continue

        # Hit a blank line: end the current paragraph, then count the run of
        # blank lines so paragraph separation is preserved (never collapsed
        # to nothing, and never so wide it breaks the page).
        flush()
        run = 0
        while index < len(lines) and not lines[index]:
            run += 1
            index += 1
        if out:
            out.append("\n" * min(2, run + 1))

    flush()

    # Drop any blank separators that ended up leading or trailing.
    return "".join(out).strip()
