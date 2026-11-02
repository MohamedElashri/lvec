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
sys.path.insert(0, str(ROOT / "docs"))
import lvec
import lvec.awkward
from _api import CLASSES, public_members, render, signature


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
        for kind, name, signature_text in re.findall(r"^```\{py:(\w+)\} ([\w.]+)(\([^\n]*\))?$", text, re.M):
            parts = name.split(".")
            value = importlib.import_module(parts[0])
            for part in parts[1:]:
                value = getattr(value, part)
            if kind == "property":
                assert isinstance(value, property), (path, name)
            if kind == "exception":
                assert isinstance(value, type) and issubclass(value, Exception), (path, name)
            if not signature_text:
                continue
            documented = ast.parse(f"def signature{signature_text}: pass").body[0].args
            actual = ast.parse(f"def signature{inspect.signature(value)}: pass").body[0].args
            if actual.args and actual.args[0].arg in ("self", "cls"):
                actual.args.pop(0)
            assert ast.dump(documented) == ast.dump(actual), (path, name, signature_text, inspect.signature(value))
            signatures += 1
    return examples, signatures


def public_api_coverage(directory):
    pages = {p.resolve(): Links(p.read_text()) for p in directory.rglob("*.html")}
    all_ids = set().union(*(page.ids for page in pages.values()))
    expected = {f"lvec.{name}" for name in lvec.__all__}
    expected |= {"lvec.Frame.name", "lvec.awkward.Array", "lvec.awkward.zip",
                 "lvec.awkward.register_awkward", "lvec.awkward.behavior"}
    expected |= {f"lvec.exceptions.{name}" for name in
                 ("LVecError", "ShapeError", "InputError", "BackendError", "DependencyError")}
    members = 0
    for docname, (name, cls) in CLASSES.items():
        # Rendering fails if any new public member has no authored definition.
        content = render(ROOT, docname, "")
        page = pages[(directory / f"{docname}.html").resolve()]
        for member in public_members(cls):
            target = f"lvec.{name}.{member}"
            assert target in page.ids, (docname, target)
            assert target in content, (docname, target)
            expected.add(target)
            if callable(getattr(cls, member)):
                signature(cls, member)
            members += 1
    assert not (expected - all_ids), f"Missing public reference objects: {sorted(expected - all_ids)}"
    return members, len(expected)


if __name__ == "__main__":
    links = html_links(Path(sys.argv[1]))
    examples, signatures = source_checks()
    members, objects = public_api_coverage(Path(sys.argv[1]))
    print(f"Checked {links} built links, {examples} Python examples, and {signatures} API signatures.")
    print(f"Public API coverage: {members} class members; {objects} public objects including factories and aliases.")
