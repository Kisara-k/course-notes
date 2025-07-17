import os
from IPython.display import Markdown, display

def render(prefix):
    if isinstance(prefix, int):
        prefix = f"{prefix:02d}"

    source_dir = os.path.dirname(os.path.abspath(__file__))
    target = next((f for f in sorted(os.listdir(source_dir)) if f.startswith(prefix) and f.endswith(".md")), None)

    if not target:
        raise FileNotFoundError(f"No .md file found starting with '{prefix}' in {source_dir}")

    with open(os.path.join(source_dir, target), "r", encoding="utf-8") as f:
        content = f.read()
        content = content.replace(r'\(', '$').replace(r'\)', '$') \
                         .replace(r'\[', '$$').replace(r'\]', '$$') \
                         .replace('#### ', '##### ')

    display(Markdown(content))
