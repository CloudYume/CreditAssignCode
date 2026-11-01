# Make the checkout importable when pytest is launched without package installation.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
