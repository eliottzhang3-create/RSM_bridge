#!/usr/bin/env python3
"""Extract official final Mizar text weights and convert to Huginn 5+10x2+5 on CPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mizar_text_5_10x2_5_huginn.conversion import main

if __name__ == '__main__':
    main()
