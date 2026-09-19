"""Extract the paper's prose into plain text for rewriting.

Keeps: section headings, all paragraph text, figure/table captions, theorem and
proposition statements (they carry prose), the abstract.
Drops: equations, tabular data, proofs, the bibliography, and every LaTeX mechanic.
Numbers are substituted from sink_numbers.tex, so the text reads as it does in the PDF.
"""
import re
import sys

PAPER = "/Volumes/2TB/iclr/paper"


def read(name):
    with open(f"{PAPER}/{name}", encoding="utf-8") as fh:
        return fh.read()


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def match_brace(s, i):
    """i points at '{'; return index just past the matching '}'."""
    depth = 0
    while i < len(s):
        if s[i] == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif s[i] == "}" and s[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(s)


def load_macros(*files):
    out = {}
    for f in files:
        s = strip_comments(read(f))
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{", s):
            start = m.end() - 1
            end = match_brace(s, start)
            out[m.group(1)] = s[start + 1:end - 1]
    return out


def expand(s, macros, rounds=6):
    for _ in range(rounds):
        before = s
        for name, val in macros.items():
            s = s.replace("\\" + name + "{}", val).replace("\\" + name + " ", val + " ")
            s = re.sub(r"\\" + name + r"(?![A-Za-z])", lambda _m, v=val: v, s)
        if s == before:
            break
    return s


def drop_environments(s, envs, marker=None):
    for env in envs:
        pat = re.compile(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}", re.S)
        s = pat.sub(lambda m: f"\n[{marker or env} omitted]\n" if marker else "\n", s)
    return s


def captions_from(block):
    r"""Pull \caption{...} text out of a float before the float is dropped."""
    out = []
    for m in re.finditer(r"\\caption\{", block):
        start = m.end() - 1
        out.append(block[start + 1:match_brace(block, start) - 1])
    return out


def keep_captions(s):
    """Replace each float with a line holding just its caption."""
    def repl(m):
        caps = captions_from(m.group(0))
        return "\n" + "\n".join(f"[{m.group(1).upper()} CAPTION] {c}" for c in caps) + "\n"
    return re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", repl, s, flags=re.S)


MATH = {
    r"\defect": "δ", r"\delta": "δ", r"\Delta": "Δ", r"\epsilon": "ε", r"\alpha": "α",
    r"\beta": "β", r"\rho": "ρ", r"\pi": "π", r"\bar m": "m̄", r"\bar a": "ā",
    r"\bar\pi": "π̄", r"\bar N": "N̄", r"\stw": "S", r"\dgm": "dgm", r"\VR": "VR",
    r"\le": "≤", r"\ge": "≥", r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\times": "×", r"\pm": "±", r"\cdot": "·", r"\ldots": "…", r"\dots": "…",
    r"\to": "→", r"\rightarrow": "→", r"\infty": "∞", r"\emptyset": "∅", r"\in": "∈",
    r"\subseteq": "⊆", r"\sum": "Σ", r"\max": "max", r"\min": "min", r"\oplus": "⊕",
    r"\mid": "|", r"\langle": "⟨", r"\rangle": "⟩", r"\equiv": "≡", r"\zeta": "ζ",
    r"\mathbb{E}": "E", r"\mathbb{R}": "R", r"\tfrac": "", r"\frac": "", r"\sqrt": "sqrt",
    r"\top": "T", r"\sigma": "σ", r"\lambda": "λ", r"\gamma": "γ", r"\mu": "μ",
    r"\tau": "τ", r"\theta": "θ", r"\phi": "φ", r"\ell": "l", r"\circ": "∘",
    r"\mathbb{F}": "F", r"\cup": "∪", r"\cap": "∩", r"\setminus": "\\",
}

# inline math kept verbatim only when it reads as a plain symbol; anything longer
# becomes a marker, since the point of this file is prose the author will rewrite
SIMPLE = re.compile(r"^[A-Za-zΑ-Ωα-ω0-9δεπρστθφλγμ_^{}()\[\].,;:+\-–=<>≤≥±×·≈/'\s|]{1,22}$")


def clean_math(m):
    body = m.group(1)
    for k, v in MATH.items():
        body = body.replace(k, v)
    body = re.sub(r"\\(?:emph|text|mathrm|mathcal|textsc|operatorname)\{([^{}]*)\}", r"\1", body)
    body = body.replace("{", "").replace("}", "").replace("\\", "")
    body = re.sub(r"\s+", " ", body).strip()
    return body if SIMPLE.match(body) else "[math]"


def clean(s):
    s = strip_comments(s)
    s = keep_captions(s)
    s = drop_environments(s, ["proof"], marker="proof")
    s = drop_environments(s, ["tabular", "longtable", "equation", "align", "itemize_raw"])
    s = re.sub(r"\\\[.*?\\\]", " [equation] ", s, flags=re.S)
    s = re.sub(r"\$\$(.*?)\$\$", " [equation] ", s, flags=re.S)
    s = re.sub(r"\$(.+?)\$", clean_math, s, flags=re.S)

    # headings
    s = re.sub(r"\\section\*?\{([^{}]*)\}", r"\n\n## \1\n", s)
    s = re.sub(r"\\subsection\*?\{([^{}]*)\}", r"\n\n### \1\n", s)
    s = re.sub(r"\\paragraph\{([^{}]*)\}", r"\n\n[\1]\n", s)
    for env, label in (("theorem", "THEOREM"), ("proposition", "PROPOSITION"),
                       ("corollary", "COROLLARY"), ("lemma", "LEMMA"), ("fact", "FACT"),
                       ("definition", "DEFINITION"), ("remark", "REMARK")):
        s = re.sub(r"\\begin\{" + env + r"\}(\[[^\]]*\])?", f"\n[{label}] ", s)
        s = re.sub(r"\\end\{" + env + r"\}", "\n", s)
    s = re.sub(r"\\begin\{(abstract)\}", "\n\n== ABSTRACT ==\n", s)
    s = re.sub(r"\\end\{abstract\}", "\n", s)

    # citations and references
    s = re.sub(r"\\cite[a-z]*\s*(\[[^\]]*\])?\s*(\[[^\]]*\])?\{([^{}]*)\}", r"[cite: \3]", s)
    s = re.sub(r"\\(?:ref|eqref|autoref)\{([^{}]*)\}", r"[ref]", s)
    s = re.sub(r"\\url\{([^{}]*)\}", r"\1", s)

    # text-level commands
    for cmd in ("emph", "textbf", "textit", "texttt", "textsc", "text", "mbox"):
        for _ in range(3):
            s = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:begin|end)\{[^{}]*\}", "", s)
    s = re.sub(r"\\(?:noindent|centering|small|tiny|normalsize|linewidth|toprule|midrule|bottomrule|newpage|appendix|maketitle|bibliography[a-z]*|input|label|setlength|resizebox|includegraphics|vspace|hspace|clearpage|iclrfinalcopy)\b\s*(\{[^{}]*\})*", "", s)
    s = re.sub(r"\\[A-Za-z]+\*?", "", s)                       # any remaining command
    s = re.sub(r"\\item\b", "\n- ", s)
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    s = re.sub(r"\[(?:label=[^\]]*|leftmargin=[^\]]*)\]", "", s)
    s = s.replace("~", " ").replace("\\\\", "\n").replace("\\ ", " ")
    s = s.replace("``", '"').replace("''", '"').replace("---", "—").replace("--", "–")
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def main():
    macros = load_macros("sink_numbers.tex", "sink_results.tex", "sink_dtype.tex",
                         "sink_precision.tex", "sink_timing.tex")
    body = read("paper.tex")
    body = body[body.index("\\begin{document}"):body.index("\\end{document}")]
    body = body.replace("\\input{appendix_sink.tex}", read("appendix_sink.tex"))
    body = re.sub(r"\\input\{[^{}]*\}", "", body)              # tables, macros, figures
    body = expand(body, macros)
    text = clean(body)

    title = re.search(r"\\title\{([^{}]*)\}", read("paper.tex"))
    header = ((f"TITLE: {title.group(1)}\n\n" if title else "") + "THE TOPOLOGY IS THE SINK — prose extracted for rewriting\n"
              "Tables, equations, proofs and the bibliography are omitted; their places are marked.\n"
              "Numbers are the generated values, exactly as they appear in the PDF.\n"
              + "=" * 78 + "\n\n")
    out = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/2TB/iclr/paper_text_for_rewrite.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(header + text + "\n")
    print(f"wrote {out}: {len(text.split())} words, {text.count(chr(10)) + 1} lines")


if __name__ == "__main__":
    main()
