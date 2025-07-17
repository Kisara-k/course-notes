import os
import re
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from nbclient import NotebookClient
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SOURCE_DIR)
RENDER_PATH = os.path.join(SOURCE_DIR, "render.py")

# 1. Create render.py if not exists
if not os.path.exists(RENDER_PATH):
    print("render.py not found. Creating it...")
    render_code = '''import os
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
        content = content.replace(r'\\(', '$').replace(r'\\)', '$') \\
                         .replace(r'\\[', '$$').replace(r'\\]', '$$') \\
                         .replace('#### ', '##### ')

    display(Markdown(content))
'''
    with open(RENDER_PATH, "w", encoding="utf-8") as f:
        f.write(render_code)
    print("render.py created.")
else:
    print("render.py already exists. Skipping creation.")


# 2. Function to process each .md file
def process_md_file(filename):
    try:
        print(f"[Thread] Processing {filename}")
        base_name = os.path.splitext(filename)[0]
        number_match = re.match(r"^(\d{2})", base_name)
        number = int(number_match.group(1)) if number_match else None

        if number is None:
            return f"Skipped {filename}: No leading number"

        notebook_name = f"{base_name}.ipynb"
        notebook_path = os.path.join(ROOT_DIR, notebook_name)

        nb = new_notebook()
        nb.cells.append(new_code_cell(f"from source.render import render\nrender({number})"))

        client = NotebookClient(nb, timeout=60, kernel_name='python3', resources={"metadata": {"path": ROOT_DIR}})
        client.execute()

        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return f"Created and executed: {notebook_name}"
    except Exception as e:
        return f"Error processing {filename}: {e}"


# 3. Run with ThreadPoolExecutor
md_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".md") and re.match(r"^\d{2}", f)]

if not md_files:
    print("No matching .md files found.")
else:
    print(f"Found {len(md_files)} .md files to process...")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_md_file, f) for f in md_files]

        for future in as_completed(futures):
            print(future.result())
