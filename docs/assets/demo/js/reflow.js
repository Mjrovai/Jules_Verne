/* ==========================================================================
   Rejoining hard-wrapped lines — the JavaScript twin of vernebot/text.py.

   The novels are Project Gutenberg plain-text editions, which break prose lines
   at roughly 67 characters and separate paragraphs with a blank line. A
   character-level model learns that faithfully, so its raw output breaks
   mid-sentence:

       He recalled upon the coast, though only at the same difficulty chattered
       with packeter round the Seas of Torres Snowy. It was the reason why the

   A single newline is a soft wrap and becomes a space; a run of two or more is
   a paragraph break and survives. Only whitespace moves -- no words change --
   which is why the feature can be toggled without altering what the model
   actually produced.

   Kept in step with the Python version by tests/check_js_parity.mjs, which runs
   both over the same inputs.
   ========================================================================== */

export function reflow(text) {
  if (!text) return text;

  const normalised = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  // Collapse runs of spaces within each line, as the Python version does.
  const lines = normalised.split('\n').map((line) => line.split(/\s+/).filter(Boolean).join(' '));

  const out = [];
  let buffer = [];

  const flush = () => {
    if (buffer.length) {
      out.push(buffer.join(' '));
      buffer = [];
    }
  };

  let i = 0;
  while (i < lines.length) {
    if (lines[i]) {
      buffer.push(lines[i]);
      i += 1;
      continue;
    }

    // Blank line: close the paragraph, then count the run so the separation is
    // preserved without ever becoming a wall of blank lines.
    flush();
    let run = 0;
    while (i < lines.length && !lines[i]) {
      run += 1;
      i += 1;
    }
    if (out.length) out.push('\n'.repeat(Math.min(2, run + 1)));
  }

  flush();
  return out.join('').trim();
}
