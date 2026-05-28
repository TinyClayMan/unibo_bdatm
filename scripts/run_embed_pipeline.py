"""Top-level entry point so the package can be invoked as a plain script.

This is equivalent to:
    PYTHONPATH=scripts python -m unified_embedding_extractor ...
but you can just point at this file:
    python scripts/run_embed_pipeline.py --mode all --encoder gearnet ...
"""

import os
import sys

# Make `unified_embedding_extractor` importable when this file is run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from unified_embedding_extractor.cli import main

if __name__ == "__main__":
    main()
