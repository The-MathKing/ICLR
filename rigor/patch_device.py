#!/usr/bin/env python3
"""
patch_device.py -- port the repository's extraction scripts from MPS-only to
CUDA-first, and drop the hardcoded macOS HuggingFace cache path.

Run from the repository root:

    python rigor/patch_device.py --dry-run     # show what would change
    python rigor/patch_device.py               # apply (writes <file>.bak)
    python rigor/patch_device.py --revert      # restore every .bak

Three substitutions, each either an exact literal or an indentation-preserving
regex. Nothing is guessed: a file that does not match a pattern is left alone
and reported as untouched.
"""
import argparse
import io
import os
import re
import sys

SKIP_DIRS = {"rigor", "archive", ".git", "refrences", "__pycache__", ".github"}

# --- 1. hardcoded mac cache -> respect the environment -----------------------
HF_OLD = 'os.environ["HF_HOME"] = "/Volumes/2TB/hf_cache"'
HF_NEW = ('# HF_HOME intentionally not set here: honour the environment.\n'
          '# (was hardcoded to "/Volumes/2TB/hf_cache", an external drive on the\n'
          '#  authors\' Mac, which does not exist on other machines.)')

# --- 2. MPS-only device selection -> CUDA first ------------------------------
DEV_OLD = 'device = "mps" if torch.backends.mps.is_available() else "cpu"'
DEV_NEW = ('device = ("cuda" if torch.cuda.is_available()\n'
           '          else "mps" if torch.backends.mps.is_available() else "cpu")')

# --- 3. mps-only cache clearing -> also clear CUDA (indentation preserved) ----
CACHE_RE = re.compile(
    r'(?P<indent>[ \t]*)if device == "mps":\n'
    r'(?P<body_indent>[ \t]+)torch\.mps\.empty_cache\(\)'
)


def cache_sub(m):
    ind, bind = m.group("indent"), m.group("body_indent")
    return (f'{ind}if device == "mps":\n'
            f'{bind}torch.mps.empty_cache()\n'
            f'{ind}elif device == "cuda":\n'
            f'{bind}torch.cuda.empty_cache()')


def iter_py(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def patch_text(src):
    hits = []
    out = src
    if HF_OLD in out:
        n = out.count(HF_OLD)
        out = out.replace(HF_OLD, HF_NEW)
        hits.append(f"HF_HOME hardcode ({n}x)")
    if DEV_OLD in out:
        n = out.count(DEV_OLD)
        out = out.replace(DEV_OLD, DEV_NEW)
        hits.append(f"mps-only device selection ({n}x)")
    out, n_cache = CACHE_RE.subn(cache_sub, out)
    if n_cache:
        hits.append(f"mps-only empty_cache ({n_cache}x)")
    return out, hits


def revert(root):
    n = 0
    for path in iter_py(root):
        bak = path + ".bak"
        if os.path.exists(bak):
            io.open(path, "w", encoding="utf-8", newline="\n").write(
                io.open(bak, encoding="utf-8").read())
            os.remove(bak)
            n += 1
    print(f"reverted {n} file(s) from .bak")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        return revert(args.root)

    changed, untouched, broken = [], 0, []
    for path in iter_py(args.root):
        src = io.open(path, encoding="utf-8").read()
        out, hits = patch_text(src)
        if out == src:
            untouched += 1
            continue
        # never write a file we just broke
        try:
            compile(out, path, "exec")
        except SyntaxError as exc:
            broken.append((path, exc))
            continue
        changed.append((path, hits))
        if not args.dry_run:
            io.open(path + ".bak", "w", encoding="utf-8", newline="\n").write(src)
            io.open(path, "w", encoding="utf-8", newline="\n").write(out)

    verb = "would patch" if args.dry_run else "patched"
    print(f"{verb} {len(changed)} file(s):")
    for path, hits in changed:
        print(f"  {os.path.relpath(path, args.root)}")
        for h in hits:
            print(f"      - {h}")
    if broken:
        print(f"\nREFUSED {len(broken)} file(s) (patch would not compile):")
        for path, exc in broken:
            print(f"  {os.path.relpath(path, args.root)}: {exc}")
    print(f"\n{untouched} file(s) needed no change.")
    if not args.dry_run and changed:
        print("Originals saved as <file>.bak  (undo: python rigor/patch_device.py --revert)")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
