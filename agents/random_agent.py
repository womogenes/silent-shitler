"""Random agent using the BaseAgent interface."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent / "shitler_env"))

from agent import SimpleRandomAgent as RandomAgent

# For backwards compatibility
__all__ = ["RandomAgent"]