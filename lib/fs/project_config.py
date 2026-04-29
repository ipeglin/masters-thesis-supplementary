import os
import sys
from pathlib import Path


def init_environment():
    """
    Detects repo root relative to this file, sets up sys.path, 
    and configures Matplotlib cache.
    """
    # This file is at: [REPO_ROOT]/lib/fs/project_config.py
    # .parents[2] climbs up: fs -> lib -> REPO_ROOT
    root = Path(__file__).resolve().parents[2]

    # 1. Add root to sys.path so we can import other local modules
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # 2. Setup Matplotlib cache
    matplotlib_cache_dir = root / ".cache" / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Crucial: This must be set BEFORE matplotlib is imported elsewhere
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

    return root

# Execute on import so the environment is ready immediately
REPO_ROOT = init_environment()