#!/usr/bin/env python3
"""Extract official final Mizar text weights and convert to 7+14x2+7 with five-slot MeSH on CPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mizar_text_7_14x2_7_mesh.conversion import main

if __name__ == '__main__':
    main()
