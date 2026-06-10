#!/usr/bin/env python3
"""Build static showcase site for LeetGPU challenges."""

import ast
import re
import shutil
from pathlib import Path

from jinja2 import Template

ROOT = Path(__file__).resolve().parent.parent
CHALLENGES_DIR = ROOT / "challenges"
OUTPUT_DIR = ROOT / "site"

FRAMEWORKS = [
    ("CUDA", ".cu", "cpp"),
    ("CUTLASS", ".cutlass.cu", "cpp"),
    ("PyTorch", ".pytorch.py", "python"),
    ("Triton", ".triton.py", "python"),
    ("JAX", ".jax.py", "python"),
    ("CuTe", ".cute.py", "python"),
    ("TileLang", ".tilelang.py", "python"),
    ("Mojo", ".mojo", "python"),
]


def extract_metadata(challenge_py: Path) -> dict:
    """Extract challenge name from challenge.py using AST."""
    with open(challenge_py) as f:
        tree = ast.parse(f.read())

    meta = {"name": None}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Challenge":
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id in meta:
                            if isinstance(item.value, ast.Constant):
                                meta[target.id] = item.value.value
    return meta


def get_number_from_dirname(name: str) -> int:
    match = re.match(r"(\d+)_", name)
    return int(match.group(1)) if match else 0


def read_challenge_html(path: Path) -> str:
    if not path.exists():
        return ""
    content = path.read_text()
    # Strip MathJax CDN script (we add it in the page template)
    content = re.sub(
        r'<script\s+src="https://cdn\.jsdelivr\.net/npm/mathjax@3'
        r'/es5/tex-mml-chtml\.js"[^>]*>\s*</script>\s*',
        "",
        content,
    )
    return content.strip()


def get_framework_files(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    files = []
    for display_name, ext, lang in FRAMEWORKS:
        if ext == ".cu":
            # Exclude .cutlass.cu files from the CUDA glob
            matched = [f for f in directory.glob(f"*{ext}") if not f.name.endswith(".cutlass.cu")]
        else:
            matched = list(directory.glob(f"*{ext}"))
        if matched:
            content = matched[0].read_text()
            files.append(
                {
                    "name": display_name,
                    "language": lang,
                    "content": content,
                    "filename": matched[0].name,
                }
            )
    return files


def extract_challenges() -> list[dict]:
    challenges = []
    for difficulty in ["easy", "medium", "hard"]:
        diff_dir = CHALLENGES_DIR / difficulty
        if not diff_dir.exists():
            continue
        for entry in sorted(diff_dir.iterdir()):
            if not entry.is_dir():
                continue
            challenge_py = entry / "challenge.py"
            challenge_html_file = entry / "challenge.html"
            if not challenge_py.exists():
                continue

            meta = extract_metadata(challenge_py)
            if not meta["name"]:
                parts = entry.name.split("_")[1:]
                meta["name"] = " ".join(w.capitalize() for w in parts)
            meta["number"] = get_number_from_dirname(entry.name)
            meta["difficulty"] = difficulty
            meta["slug"] = entry.name

            html_content = read_challenge_html(challenge_html_file)
            meta["html_content"] = html_content

            snippet = re.sub(r"<[^>]+>", "", html_content)
            snippet = re.sub(r"\s+", " ", snippet).strip()[:200]
            meta["snippet"] = snippet

            meta["solution_files"] = get_framework_files(entry / "solution")

            challenges.append(meta)
    return challenges


TEMPLATES_DIR = ROOT / "scripts" / "templates"


def build_site():
    challenges = extract_challenges()
    challenges.sort(
        key=lambda c: ({"easy": 0, "medium": 1, "hard": 2}[c["difficulty"]], c["number"])
    )

    total = len(challenges)
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for c in challenges:
        counts[c["difficulty"]] += 1

    # Compute framework data for homepage filter
    all_frameworks = sorted(set(f["name"] for c in challenges for f in c["solution_files"]))
    fw_counts = {fw: 0 for fw in all_frameworks}
    for c in challenges:
        frameworks = [f["name"] for f in c["solution_files"]]
        c["framework_attr"] = ",".join(frameworks)
        for fw in frameworks:
            fw_counts[fw] += 1

    # Clean output dir
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # Copy assets
    assets_src = ROOT / "assets"
    assets_dst = OUTPUT_DIR / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, assets_dst)

    # Render index
    index_tpl = Template((TEMPLATES_DIR / "index.html").read_text())
    html = index_tpl.render(
        challenges=challenges,
        total=total,
        counts=counts,
        all_frameworks=all_frameworks,
        fw_counts=fw_counts,
        root=".",
    )
    (OUTPUT_DIR / "index.html").write_text(html)

    # Render detail pages
    detail_tpl = Template((TEMPLATES_DIR / "detail.html").read_text())
    for i, c in enumerate(challenges):
        prev_c = challenges[i - 1] if i > 0 else None
        next_c = challenges[i + 1] if i < len(challenges) - 1 else None

        page_dir = OUTPUT_DIR / "challenges" / c["difficulty"] / c["slug"]
        page_dir.mkdir(parents=True, exist_ok=True)

        html = detail_tpl.render(c=c, prev=prev_c, next=next_c, root="../../..")
        (page_dir / "index.html").write_text(html)

    print(f"Built {total} challenge pages → {OUTPUT_DIR}")


if __name__ == "__main__":
    build_site()
