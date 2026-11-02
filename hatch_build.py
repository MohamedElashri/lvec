"""Keep repository navigation relative and package-description links absolute."""
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

from hatchling.metadata.plugin.interface import MetadataHookInterface


class ReadmeMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        text = (Path(self.root) / "README.md").read_text(encoding="utf-8")
        base = "https://github.com/MohamedElashri/lvec/blob/main/"

        def resolve(match):
            target = match.group(1)
            parsed = urlsplit(target)
            if target.startswith("#") or parsed.scheme or parsed.netloc:
                return target
            return urljoin(base, target)

        text = re.sub(r"(?<=\]\()([^\s)]+)(?=\))", resolve, text)
        metadata["readme"] = {"content-type": "text/markdown", "text": text}
