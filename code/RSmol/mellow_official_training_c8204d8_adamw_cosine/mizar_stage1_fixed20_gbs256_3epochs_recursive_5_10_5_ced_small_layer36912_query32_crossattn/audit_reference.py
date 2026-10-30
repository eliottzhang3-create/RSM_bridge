"""Source audit for this isolated route; no remote PASS claim."""
from audit_parent import *

if __name__ == "__main__":
    import json
    from contract import write_json
    result = audit()
    write_json(ROOT / "reference_audit.json", result)
    print(json.dumps(result, indent=2))
