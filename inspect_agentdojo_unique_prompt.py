import json
import re

in_path = "inspect_agentdojo_export.jsonl"
out_path = "inspect_agentdojo_export_i0.jsonl"

pat = re.compile(r"-i0$")  # name ends with "...-i0"

kept = 0
total = 0

with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        total += 1

        obj = json.loads(line)
        name = obj.get("name", "")
        if pat.search(name):
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

print(f"Read {total} examples from {in_path}")
print(f"Wrote {kept} examples to {out_path}")