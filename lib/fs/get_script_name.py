import inspect
import os
import sys
from pathlib import Path


def get_name():
    # 1. Check for standard .py script first
    stack = inspect.stack()
    # Search the stack for the first file that isn't THIS file
    this_file = Path(__file__).resolve()
    caller_path = None
    for frame in stack:
        f_path = Path(frame.filename).resolve()
        if f_path != this_file:
            caller_path = f_path
            break
            
    # If it's a real file on disk (not a temporary Jupyter input), return it
    if caller_path and caller_path.exists() and caller_path.suffix in ['.py', '.ipynb']:
        return caller_path.stem

    # 2. Handle Jupyter/IPython specifically
    try:
        import __main__

        # VS Code sets this specifically
        vsc_file = getattr(__main__, '__vsc_ipynb_file__', None)
        if vsc_file:
            return Path(vsc_file).stem
            
        # 3. Fallback for JupyterLab/Notebook
        # Some environments store the path in __session__
        session = getattr(__main__, '__session__', None)
        if session and not session.isdigit(): # Ignore numeric session IDs
            return Path(session).stem
            
    except Exception:
        pass

    return "notebook"
