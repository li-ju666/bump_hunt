"""Build the self-contained blog post -> site/index.html.

    uv run python scripts/export_interactive.py --run runs/b30000_s0   # precompute the data
    uv run python scripts/build_site.py                                 # docs/post.md -> site/index.html

The prose lives in docs/post.md (Markdown with $...$ and $$...$$ math). The line
`<!-- explorer -->` marks where the interactive section goes. scripts/site_template.html
holds the page shell, styles, the explorer and its script.

The result is one HTML file (data inline; Plotly, KaTeX and fonts from CDNs) that can be
served by GitHub Pages: push the repository and set Pages to the `site/` folder.
"""

import re
from pathlib import Path

import markdown

root = Path(__file__).resolve().parent.parent
MARK = "<!-- explorer -->"


def md_to_html(text):
    """Markdown -> HTML, with math and fenced code protected from Markdown's own parsing."""
    stash = []

    def keep(s):
        stash.append(s)
        return f"QQSTASH{len(stash) - 1}QQ"

    text = re.sub(r"```.*?```", lambda m: keep(m.group(0)), text, flags=re.S)       # code first
    text = re.sub(r"\$\$.*?\$\$", lambda m: keep(m.group(0)), text, flags=re.S)     # display math
    text = re.sub(r"(?<!\\)\$[^\n$]+?\$", lambda m: keep(m.group(0)), text)         # inline math
    text = re.sub(r"QQSTASH(\d+)QQ", lambda m: stash[int(m.group(1))] if stash[int(m.group(1))].startswith("```") else m.group(0), text)
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    return re.sub(r"QQSTASH(\d+)QQ", lambda m: stash[int(m.group(1))], html)


post = (root / "docs/post.md").read_text()
assert MARK in post, f"{MARK} missing from docs/post.md"
for frag in ("mechanism", "architecture"):   # <!-- name --> in post.md -> scripts/name.html
    post = post.replace(f"<!-- {frag} -->", (root / f"scripts/{frag}.html").read_text())
top, bottom = post.split(MARK, 1)
data = (root / "results/interactive_data.json").read_text().replace("</", "<\\/")
tpl = (root / "scripts/site_template.html").read_text()
for ph in ("/*__ARTICLE_TOP__*/", "/*__ARTICLE_BOTTOM__*/", "/*__DATA__*/"):
    assert ph in tpl, f"{ph} missing from template"
page = (tpl.replace("/*__ARTICLE_TOP__*/", md_to_html(top))
           .replace("/*__ARTICLE_BOTTOM__*/", md_to_html(bottom))
           .replace("/*__DATA__*/", data))
out = root / "site/index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page)
print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
