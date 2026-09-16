#!/usr/bin/env python3
"""Structural verification of paper.tex."""
import re, os, sys, collections

B = sys.argv[1] if len(sys.argv) > 1 else "paper"
main = os.path.join(B, "paper.tex")
tex = "".join(open(os.path.join(B, f), encoding="utf-8").read()
              for f in ("paper.tex", "related_work.tex"))

labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
beg = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", tex))
end = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", tex))

print("ref-without-label :", sorted(refs - labels) or "none")
print("unbalanced envs   :", [(k, beg[k], end[k]) for k in set(beg) | set(end)
                              if beg[k] != end[k]] or "none")
print("brace delta       :", tex.count("{") - tex.count("}"))
print("$ parity          :", tex.count("$") % 2)
print("tables / figures  :", beg['table'] + beg['table*'], "/", beg['figure'] + beg['figure*'])

# citation keys
bib = open(os.path.join(B, "paper.bib"), encoding="utf-8").read()
defined = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for m in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
    cited.update(k.strip() for k in m.split(","))
print("cited-not-in-bib  :", sorted(cited - defined) or "none")

# tabular column counts
lines = open(main, encoding="utf-8").read().split("\n")
spec, want, bad = None, 0, 0
for i, l in enumerate(lines, 1):
    m = re.search(r"\\begin\{tabular\}\{([lcrp@{}\\.0-9|]+)\}", l)
    if m:
        spec = m.group(1); want = len(re.findall(r"[lcrp]", spec)); continue
    if re.search(r"\\end\{tabular\}", l):
        spec = None; continue
    if spec and l.rstrip().endswith("\\\\") and "multicolumn" not in l and "cmidrule" not in l:
        got = l.count("&") + 1
        if got != want:
            bad += 1
            print(f"  COLUMN MISMATCH line {i}: {got} cells, spec wants {want}")
print("column mismatches :", bad)

# stale strings that must be gone
stale = [
    "open diagnostic question",
    "this int8-quantized model at this sequence-length regime",
    "16GB-RAM hardware cannot hold",
    "all six model instances",
    "less than $0.003$",
    "Best Non-Topo",
    "not confirmed for any setting",
]
print()
for pat in stale:
    n = tex.count(pat)
    print(f"  {'OK ' if n == 0 else 'STALE'}  ({n})  {pat}")
