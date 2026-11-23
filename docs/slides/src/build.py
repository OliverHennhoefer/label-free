"""Build the slide deck PDF from its source.

Source: deck.json (slide order), deck.css (style), slides/<id>.html (one
1920x1080 slide each), viewer.css (page frame used only for this build).

    python build.py            # writes deck.html next to this file
    node pdf.js ../label-free-evaluation.pdf   # needs Playwright with Chromium

Fonts (Inter, JetBrains Mono) load from Google Fonts, so the build needs
network access.
"""
import json
from pathlib import Path

R = Path(__file__).parent
PRINT = "@page{size:1920px 1080px;margin:0} body > section{break-after:page}"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;600"
    "&family=JetBrains+Mono:wght@400;600&display=swap"
)

deck = json.loads((R / "deck.json").read_text(encoding="utf-8"))
styles = "".join(
    f"<style>{s}</style>"
    for s in (
        (R / "viewer.css").read_text(encoding="utf-8"),
        (R / "deck.css").read_text(encoding="utf-8"),
        PRINT,
    )
)
html = (
    '<!doctype html><html><head><meta charset="utf-8">'
    f"<title>{deck['title']}</title>"
    f'<link rel="stylesheet" href="{FONTS}">{styles}</head><body>'
)
for slide in deck["order"]:
    html += (R / "slides" / f"{slide}.html").read_text(encoding="utf-8")
(R / "deck.html").write_text(html + "</body></html>", encoding="utf-8")
