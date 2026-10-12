"""Execute with this Python's kernel, save outputs, and export an HTML preview."""
from pathlib import Path
import argparse
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager


def execute(notebook_name="Fair_Aid_QAOA.ipynb"):
    directory = Path(__file__).resolve().parent
    runtime = directory.parents[1] / "downloads" / "track2" / "jupyter-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    os.environ["PYTHONUTF8"] = "1"
    path = directory / notebook_name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    print(f"Executing {path.name} using {sys.executable}", flush=True)
    try:
        NotebookClient(notebook, km=manager, timeout=600,
                       resources={"metadata": {"path": str(directory)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    errors = [out for cell in notebook.cells if cell.cell_type == "code"
              for out in cell.get("outputs", []) if out.output_type == "error"]
    if errors:
        raise RuntimeError("Notebook contains error outputs.")
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    exporter = HTMLExporter(template_name="lab")
    html, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": path.stem.replace("_", " ")}})
    preview = path.with_suffix(".html")
    preview.write_text(html, encoding="utf-8")
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Saved {count} executed code cells; preview: {preview}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="Fair_Aid_QAOA.ipynb")
    execute(parser.parse_args().notebook)
