"""Prompt files live in prompts/*.txt and use string.Template $placeholders."""
import hashlib
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> Template:
    return Template((PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8"))


def prompt_versions() -> dict[str, str]:
    return {p.name: hashlib.sha1(p.read_bytes()).hexdigest()[:12]
            for p in sorted(PROMPT_DIR.glob("*")) if p.is_file()}


def render_anchors() -> str:
    import yaml
    data = yaml.safe_load((PROMPT_DIR / "anchors.yaml").read_text(encoding="utf-8"))
    out = [f"SCORING RULE: {data['rule'].strip()}"]
    for key, label in (("hooks", "HOOK"), ("cliffhangers", "CLIFFHANGER")):
        out.append(f"{label} SCORE ANCHORS:")
        out += [f"EXAMPLE (score {a['score']}/10): {a['example']} — {a['why']}" for a in data[key]]
    return "\n".join(out)
