"""Unit test conftest — inserts packages/ into sys.path before test collection."""
import os
import sys

# Insert packages/ at position 0 so summit packages can be found.
_packages_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "packages")
)
if _packages_dir not in sys.path:
    sys.path.insert(0, _packages_dir)
