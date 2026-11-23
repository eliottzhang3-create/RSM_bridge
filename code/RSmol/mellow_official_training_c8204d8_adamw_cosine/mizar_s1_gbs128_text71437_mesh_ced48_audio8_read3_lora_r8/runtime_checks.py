"""Optional synthetic CPU/autograd checks; not a formal-submission prerequisite."""
import sys
import unittest
from pathlib import Path

if __name__ == '__main__':
    tests = Path(__file__).resolve().parents[4] / 'tests'
    suite = unittest.defaultTestLoader.discover(str(tests),
        pattern='test_mizar_s1_text71437_ced48_audio8_read3.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print(f'UNVERIFIED: {len(result.skipped)} dependency-dependent checks skipped; no GPU PASS')
    sys.exit(0 if result.wasSuccessful() else 1)
