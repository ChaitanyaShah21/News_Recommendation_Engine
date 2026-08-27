"""Render reports/design_note.md to PDF and report its page count.

Q6 caps the design note at 4 pages, which is a property of the *rendered* document,
not the Markdown. This makes that number checkable instead of guessed.

Pipeline: Markdown -> HTML (python-markdown, tables + fenced code) -> PDF (headless
Chromium). Chromium rather than wkhtmltopdf because wkhtmltopdf's engine is an old
WebKit that mis-renders the tables this note relies on.

Page count is read from the PDF's own page tree rather than estimated from word
count -- an estimate is exactly the kind of number this project has learned not to
trust (see PROGRESS.md on three wrong runtime estimates in a row).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "design_note.md"
HTML = ROOT / "reports" / "design_note.html"
PDF = ROOT / "reports" / "design_note.pdf"
LIMIT = 4

CSS = """
@page { size: A4; margin: 14mm 14mm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 9.8pt; line-height: 1.30;
       color: #111; margin: 0; }
h1 { font-size: 16pt; margin: 0 0 .4em; }
h2 { font-size: 11.5pt; margin: .85em 0 .3em; border-bottom: 1px solid #ccc;
     padding-bottom: 2px; }
h3 { font-size: 10pt; margin: .7em 0 .25em; }
p  { margin: .36em 0; }
ul, ol { margin: .4em 0; padding-left: 1.3em; }
li { margin: .15em 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt;
       background: #f4f4f4; padding: 0 2px; }
pre { background: #f6f6f6; padding: .5em .7em; font-size: 8.4pt; overflow-x: auto;
      border-left: 2px solid #ddd; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: .4em 0;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2.5px 5px; text-align: left; }
th { background: #efefef; }
img { max-width: 76%; height: auto; display: block; margin: .4em auto;
      page-break-inside: avoid; }
em { color: #444; }
blockquote { margin: .5em 0 .5em .8em; padding-left: .7em; border-left: 2px solid #ccc;
             color: #444; }
h2, h3 { page-break-after: avoid; }
"""


def find_chromium() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit("FATAL: no Chromium/Chrome on PATH; cannot render a PDF.")


def page_count(pdf: Path) -> int:
    """Read the page count out of the PDF rather than estimating it."""
    raw = pdf.read_bytes()
    m = re.search(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", raw, re.S)
    if m:
        return int(m.group(1))
    # Fallback: count page objects directly.
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", raw))


def main() -> int:
    if not SRC.exists():
        sys.exit(f"FATAL: {SRC} not found")

    import markdown

    html_body = markdown.markdown(
        SRC.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    HTML.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>",
        encoding="utf-8",
    )

    subprocess.run(
        [find_chromium(), "--headless", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True, capture_output=True,
    )

    pages = page_count(PDF)
    words = len(SRC.read_text(encoding="utf-8").split())
    print(f"words     : {words}")
    print(f"PDF       : {PDF.relative_to(ROOT)}  ({PDF.stat().st_size / 1024:.0f} KB)")
    print(f"pages     : {pages}  (limit {LIMIT})")
    if pages > LIMIT:
        print(f"OVER by {pages - LIMIT} page(s) -- trim before submitting.")
        return 1
    print("within the limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
