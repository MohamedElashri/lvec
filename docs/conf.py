"""Build the manual from Markdown and the canonical repository guides."""
import ast
import os
from pathlib import Path
import posixpath
import re
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
SOURCE_REF = os.environ.get("DOCS_SOURCE_REF", "main")
sys.path.insert(0, str(ROOT))
project = "LVec"
author = "Mohamed Elashri"
copyright = "2026, Mohamed Elashri"
module = ast.parse((ROOT / "lvec/__init__.py").read_text())
release = next(ast.literal_eval(node.value) for node in module.body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "__version__"
                       for target in node.targets))
version = release
extensions = ["myst_parser", "sphinx.ext.doctest", "sphinx.ext.mathjax"]
myst_enable_extensions = ["dollarmath", "fieldlist"]
myst_heading_anchors = 6
nitpicky = True
exclude_patterns = ["_build", "check.py"]
html_theme = "furo"
html_title = f"LVec {release}"
html_static_path = []
html_theme_options = {"source_repository": "https://github.com/MohamedElashri/lvec/",
                      "source_branch": SOURCE_REF, "source_directory": "docs/"}
html_baseurl = "https://melashri.net/lvec/"
html_last_updated_fmt = None
doctest_global_setup = "import warnings; warnings.simplefilter('error', RuntimeWarning)"

# Keep the physics guide, measurements, and contributor instructions in one
# place while rendering them as searchable manual pages. Repository-relative
# links stay useful on GitHub; code links in the manual point to its source ref.
SHARED = {"physics": ROOT / "user-guide.md",
          "benchmarks": ROOT / "benchmarks/README.md",
          "development": ROOT / "CONTRIBUTING.md"}


def manual_links(app, docname, source):
    path = SHARED.get(docname, ROOT / "docs" / f"{docname}.md")
    if not path.exists():
        return
    if docname in SHARED:
        app.env.note_dependency(str(path))
    pages = {value.resolve(): key for key, value in SHARED.items()}
    pages[ROOT / "README.md"] = "index"
    pages.update({p.resolve(): p.relative_to(ROOT / "docs").with_suffix("").as_posix()
                  for p in (ROOT / "docs").rglob("*.md")})

    def link(match):
        target = match.group(1)
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        destination = (path.parent / parts.path).resolve()
        anchor = f"#{parts.fragment}" if parts.fragment else ""
        if destination in pages:
            relative = posixpath.relpath(f"{pages[destination]}.md", posixpath.dirname(docname) or ".")
            return f"]({relative}{anchor})"
        relative = destination.relative_to(ROOT).as_posix()
        kind = "tree" if destination.is_dir() else "blob"
        return f"](https://github.com/MohamedElashri/lvec/{kind}/{SOURCE_REF}/{relative}{anchor})"

    content = path.read_text() if docname in SHARED else source[0]
    source[0] = re.sub(r"\]\(([^\s)]+)\)", link, content)


def canonical_source(app, pagename, templatename, context, doctree):
    if pagename in SHARED:
        path = SHARED[pagename].relative_to(ROOT).as_posix()
        base = "https://github.com/MohamedElashri/lvec"
        context["theme_source_view_link"] = f"{base}/blob/{SOURCE_REF}/{path}?plain=true"
        context["theme_source_edit_link"] = f"{base}/edit/{SOURCE_REF}/{path}"


def setup(app):
    app.connect("source-read", manual_links)
    app.connect("html-page-context", canonical_source)
