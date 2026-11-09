#!/usr/bin/env python3
"""Convert final Mizar text weights to 5+10x2+5 with per-loop rank8 LoRA on CPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mizar_text_5_10x2_5_loop_lora_r8.conversion import main

if __name__ == '__main__':
    main()
