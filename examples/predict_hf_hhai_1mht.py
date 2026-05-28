#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

from esmfold2_predict import main


if __name__ == "__main__":
    default_input = Path(__file__).resolve().parent / "af3_inputs" / "hf_hhai_1mht.json"
    sys.argv[1:1] = ["--input", str(default_input)]
    main()
