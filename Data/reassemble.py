from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
OUT = HERE / "restored_workbooks"
OUT.mkdir(exist_ok=True)
for item in MANIFEST["files"]:
    target = OUT / item["original_name"]
    temporary = target.with_suffix(target.suffix + ".tmp")
    sha = hashlib.sha256()
    with temporary.open("wb") as dest:
        for part in item["parts"]:
            source = HERE / part["name"]
            data = source.read_bytes()
            assert len(data) == part["size_bytes"], source
            assert hashlib.sha256(data).hexdigest() == part["sha256"], source
            dest.write(data)
            sha.update(data)
    assert temporary.stat().st_size == item["size_bytes"], target
    assert sha.hexdigest() == item["sha256"], target
    temporary.replace(target)
    print("Restored and verified:", target)
