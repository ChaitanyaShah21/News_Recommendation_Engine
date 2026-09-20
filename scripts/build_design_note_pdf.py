"""Render a design note to PDF and report its page count.

Defaults to A2's note (Q6: 6-page TARGET, 11pt, 1-inch margins - a guideline, not a
cap, so going over prints a warning rather than failing). A1's note is rendered with
`--src reports/design_note.md --pdf reports/design_note.pdf --pt 9.8 --margin 14mm
--limit 4 --hard`, which is the 4-page cap it was written against.

The page count is a property of the *rendered* document, not the Markdown. This makes
that number checkable instead of guessed.

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
import argparse

CSS_TEMPLATE = """
@page { size: A4; margin: __MARGIN__; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: __PT__pt; line-height: 1.22;
       color: #111; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 .15em; letter-spacing: -0.2px; }
h2 { font-size: 11.5pt; margin: .62em 0 .26em; border-bottom: 1px solid #bbb;
     padding-bottom: 2px; }
h3 { font-size: 10.5pt; margin: .2em 0 .5em; font-weight: normal; color: #333;
     font-style: italic; }
h1 + h3 { margin-bottom: .8em; }
p  { margin: .28em 0; }
ul, ol { margin: .3em 0; padding-left: 1.2em; }
li { margin: .15em 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt;
       background: #f4f4f4; padding: 0 2px; }
pre { background: #f7f7f7; padding: .45em .6em; font-size: 7.6pt; overflow: hidden;
      white-space: pre;
      border-left: 2px solid #ddd; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: .35em 0; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 1.8px 4px; text-align: left; }
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="reports/design_note_a2.md")
    ap.add_argument("--pdf", default="reports/design_note_a2.pdf")
    ap.add_argument("--pt", default="11", help="body font size; Q6 says 11pt")
    ap.add_argument("--margin", default="25.4mm", help="Q6 says 1 inch")
    ap.add_argument("--limit", type=int, default=6, help="page TARGET (see --hard)")
    ap.add_argument("--hard", action="store_true",
                    help="treat --limit as a cap and exit non-zero when over")
    args = ap.parse_args()
    SRC, PDF = ROOT / args.src, ROOT / args.pdf
    HTML = PDF.with_suffix(".html")
    LIMIT = args.limit
    CSS = CSS_TEMPLATE.replace("__MARGIN__", args.margin).replace("__PT__", args.pt)
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
        over = pages - LIMIT
        print(f"{over} page(s) over the {LIMIT}-page target"
              + (" -- trim before submitting." if args.hard else
                 " -- Q6 allows this if the content justifies it; check for padding."))
        return 1 if args.hard else 0
    print("within the target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
