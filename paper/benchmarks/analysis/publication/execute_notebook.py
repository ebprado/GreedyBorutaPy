"""Execute this notebook with IPython, without starting a Jupyter socket server.

Useful for batch regeneration on the local Anaconda installation when ZeroMQ
cannot start a kernel. The notebook remains a normal Jupyter notebook. Run from
any working directory; only a successful full execution replaces its outputs.
"""
import os
import sys
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support
from traitlets.config import Config


def main():
    # Optional argument: notebook name in this directory (default: the diagnostics notebook).
    name = sys.argv[1] if len(sys.argv) > 1 else "publication_analysis.ipynb"
    path = Path(__file__).with_name(name).resolve()
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance(config=Config({"HistoryManager": {"hist_file": ":memory:"}}))
    configure_inline_support(shell, "module://matplotlib_inline.backend_inline")
    original_directory = Path.cwd()
    try:
        os.chdir(path.parent)
        execution = 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            execution += 1
            with capture_output(display=True) as captured:
                result = shell.run_cell(cell.source, store_history=True)
            error = result.error_before_exec or result.error_in_exec
            if error:
                print(captured.stdout)
                print(captured.stderr)
                raise RuntimeError(f"Notebook cell {index + 1} failed") from error
            cell.execution_count = execution
            cell.outputs = []
            for name in ["stdout", "stderr"]:
                value = getattr(captured, name)
                if value:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output(
                    "display_data", data=output.data, metadata=output.metadata))
            print(f"Executed cell {index + 1}", flush=True)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f"Saved {path}")
    finally:
        os.chdir(original_directory)


if __name__ == "__main__":
    main()
