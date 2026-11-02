"""Check manual links, runnable examples, and API signatures without network I/O."""
import ast
from html.parser import HTMLParser
import importlib
import inspect
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.targets = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        if tag == "a" and "href" in values:
            self.targets.append(values["href"])
        if tag in ("img", "script") and "src" in values:
            self.targets.append(values["src"])
        if tag == "link" and "href" in values:
            self.targets.append(values["href"])


def html_links(directory):
    pages = {p.resolve(): Links(p.read_text()) for p in directory.rglob("*.html")}
    assert pages, f"No built HTML found in {directory}"
    count = 0
    for path, page in pages.items():
        for target in page.targets:
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                continue
            destination = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if destination.is_dir():
                destination /= "index.html"
            assert destination.is_relative_to(directory.resolve()), (path, target)
            assert destination.exists(), (path, target)
            if parts.fragment and destination in pages:
                assert unquote(parts.fragment) in pages[destination].ids, (path, target)
            count += 1
    return count


def source_checks():
    files = [ROOT / "README.md", ROOT / "user-guide.md", ROOT / "CONTRIBUTING.md",
             ROOT / "benchmarks/README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    examples = signatures = 0
    for path in files:
        text = path.read_text()
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        prose = re.sub(r"`[^`]+`", "", prose)
        math = r"\$\$(.*?)\$\$|\$([^$\n]+)\$"
        assert "$" not in re.sub(math, "", prose, flags=re.S), path
        for match in re.finditer(math, prose, re.S):
            assert r"\operatorname" not in match.group(), path
        for code in re.findall(r"```python\n(.*?)```", text, re.S):
            result = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", code],
                                    cwd=ROOT, capture_output=True, text=True)
            assert result.returncode == 0, (path, result.stderr)
            examples += 1
        for name, signature in re.findall(r"^```\{py:\w+\} ([\w.]+)(\([^\n]*\))$", text, re.M):
            parts = name.split(".")
            value = importlib.import_module(parts[0])
            for part in parts[1:]:
                value = getattr(value, part)
            documented = ast.parse(f"def signature{signature}: pass").body[0].args
            actual = ast.parse(f"def signature{inspect.signature(value)}: pass").body[0].args
            if actual.args and actual.args[0].arg in ("self", "cls"):
                actual.args.pop(0)
            assert ast.dump(documented) == ast.dump(actual), (path, name, signature, inspect.signature(value))
            signatures += 1
    return examples, signatures


if __name__ == "__main__":
    links = html_links(Path(sys.argv[1]))
    examples, signatures = source_checks()
    print(f"Checked {links} built links, {examples} Python examples, and {signatures} API signatures.")
