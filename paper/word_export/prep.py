"""Turn v3.tex into pandoc-friendly LaTeX with all numbers resolved."""
import re, json

src = open('v3.tex', encoding='utf-8').read()
body = src[src.index('\\begin{document}') + len('\\begin{document}'):src.index('\\end{document}')]

ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
LET = 'ABCDEFGHIJKLMNOP'

# ---------- 1. numbering of sections, floats, equations ----------
labels = {}
tok = re.compile(r'\\(section|subsection|subsubsection)\{|\\begin\{(figure\*?|table\*?|equation|definition)\}'
                 r'|\\label\{([^}]+)\}|\\ALGORITHM')
sec = sub = subsub = 0
fig = tab = eq = 0
cur = None
for m in tok.finditer(body):
    if m.group(1) == 'section':
        sec += 1; sub = 0; cur = ('sec', ROMAN[sec - 1])
    elif m.group(1) == 'subsection':
        sub += 1; subsub = 0; cur = ('sec', f'{ROMAN[sec-1]}-{LET[sub-1]}')
    elif m.group(1) == 'subsubsection':
        subsub += 1; cur = ('sec', f'{ROMAN[sec-1]}-{LET[sub-1]}{subsub}')
    elif m.group(2):
        e = m.group(2).rstrip('*')
        if e == 'figure':
            fig += 1; cur = ('fig', str(fig))
        elif e == 'table':
            tab += 1; cur = ('tab', ROMAN[tab - 1])
        elif e == 'equation':
            eq += 1; cur = ('eq', str(eq))
    elif m.group(3):
        labels[m.group(3)] = cur[1]
labels['alg:main'] = '1'
print('figures', fig, 'tables', tab, 'equations', eq)

# ---------- 2. citation numbers in order of first citation ----------
order = []
for m in re.finditer(r'\\cite\{([^}]+)\}', body):
    for k in m.group(1).split(','):
        k = k.strip()
        if k not in order:
            order.append(k)
num = {k: i + 1 for i, k in enumerate(order)}
json.dump(order, open('cite_order.json', 'w'))
print('references', len(order))


def fmt_cite(m):
    ns = sorted(num[k.strip()] for k in m.group(1).split(','))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f'[{ns[i]}]--[{ns[j]}]')
        else:
            out += [f'[{n}]' for n in ns[i:j + 1]]
        i = j + 1
    return ', '.join(out)


body = re.sub(r'\\cite\{([^}]+)\}', fmt_cite, body)

# ---------- 3. references ----------
body = re.sub(r'\\eqref\{([^}]+)\}', lambda m: f'({labels[m.group(1)]})', body)
body = re.sub(r'\\ref\{([^}]+)\}', lambda m: labels[m.group(1)], body)

# ---------- 4. macros ----------
rep = {r'\pof': r'\mathrm{PoF}_t', r'\cpra': r'\mathrm{cPRA}', r'\cit': r'\mathrm{CIT}',
       r'\ilp': r'\mathrm{ILP}', r'\checkmark': '✓'}
for a, b in rep.items():
    body = re.sub(re.escape(a) + r'(?![a-zA-Z])', lambda m, b=b: b, body)
body = body.replace(r'\IEEEPARstart{K}{idney}', 'Kidney')

body = re.sub(r'\$\^\{([ab])\}\$', r'\\textsuperscript{\1}', body)

# ---------- 5. headings ----------
sec = sub = subsub = 0


def head(m):
    global sec, sub, subsub
    kind, title = m.group(1), m.group(2)
    if kind == 'section':
        sec += 1; sub = 0
        return f'\\section*{{{ROMAN[sec-1]}. {title}}}'
    if kind == 'subsection':
        sub += 1; subsub = 0
        return f'\\subsection*{{{LET[sub-1]}. {title}}}'
    subsub += 1
    return f'\\subsubsection*{{{subsub}) {title}}}'


body = re.sub(r'\\(section|subsection|subsubsection)\{((?:[^{}]|\{[^{}]*\})*)\}', head, body, flags=re.S)
body = re.sub(r'\\label\{[^}]+\}', '', body)

# ---------- 6. equations: add the number ----------
eqn = 0


def eqfix(m):
    global eqn
    eqn += 1
    inner = m.group(1).strip().rstrip()
    return f'\\[ {inner} \\] EQNO{eqn}QE '


body = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', eqfix, body, flags=re.S)

# ---------- 7. definitions ----------
dn = 0


def deffix(m):
    global dn
    dn += 1
    return f'\n\n\\emph{{Definition {dn} ({m.group(1)}):}} {m.group(2).strip()}\n\n'


body = re.sub(r'\\begin\{definition\}\[([^\]]+)\](.*?)\\end\{definition\}', deffix, body, flags=re.S)


SYM = {'pm': '±', 'Delta': 'Δ', 'tau': '\\textit{τ}', 'lambda': '\\textit{λ}', 'mu': '\\textit{μ}', 'ell': '\\textit{ℓ}', 'kappa': '\\textit{κ}',
       'alpha': '\\textit{α}', 'rho': '\\textit{ρ}', 'sim': '∼', 'dots': '…', 'geq': '≥', 'leq': '≤', 'times': '×',
       '{': '{', '}': '}', ',': ' ', ';': ' ', 'quad': ' ', 'in': '∈'}


def m2t(s):
    out, i = [], 0

    def group(i):
        if s[i] == '{':
            d, j = 1, i + 1
            while d:
                d += {'{': 1, '}': -1}.get(s[j], 0); j += 1
            return s[i + 1:j - 1], j
        if s[i] == '\\':
            m = re.match(r'\\[a-zA-Z]+', s[i:])
            return m.group(0), i + len(m.group(0))
        return s[i], i + 1
    while i < len(s):
        c = s[i]
        if c == '\\':
            m = re.match(r'\\([a-zA-Z]+|.)', s[i:])
            name = m.group(1); i += len(m.group(0))
            if name in ('mathrm', 'text', 'max', 'mathbf'):
                if name == 'max':
                    out.append('max'); continue
                g, i = group(i)
                out.append(g if name != 'mathbf' else '\\textbf{' + g + '}')
            else:
                out.append(SYM.get(name, name))
        elif c in '_^':
            g, i = group(i + 1)
            cmd = 'textsubscript' if c == '_' else 'textsuperscript'
            out.append('\\' + cmd + '{' + m2t(g) + '}')
        elif c == '-':
            out.append('−'); i += 1
        elif c == '=':
            out.append(' = '); i += 1
        elif c.isalpha():
            out.append('\\textit{' + c + '}'); i += 1
        elif c in '{}':
            i += 1
        elif c == ' ':
            i += 1
        else:
            out.append(c); i += 1
    t = ''.join(out)
    return re.sub(r' +', ' ', t)


def textmath(s):
    return re.sub(r'\$([^$]+)\$', lambda m: m2t(m.group(1)), s)

# ---------- 8. figures ----------
WID = {'fig1_anzkx.jpg': '4.3in', 'fig2_structures.jpg': '4.3in'}
fn = 0


def figfix(m):
    global fn
    fn += 1
    f = re.search(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', m.group(0)).group(1)
    cap = re.search(r'\\caption\{(.*)\}\s*$', m.group(0)[:m.group(0).rindex('\\end')].strip(), re.S).group(1)
    w = WID.get(f, '6.5in')
    cap = textmath(cap)
    return (f'\n\n\\begin{{figure}}\n\\centering\n\\includegraphics[width={w}]{{fig/{f}}}\n'
            f'\\caption{{Fig.~{fn}.~{cap}}}\n\\end{{figure}}\n\n')


body = re.sub(r'\\begin\{figure\*?\}(?:\[[^\]]*\])?(.*?)\\end\{figure\*?\}', figfix, body, flags=re.S)


# ---------- 9. tables ----------
tn = 0


def tabfix(m):
    global tn
    tn += 1
    t = m.group(1)
    cap = re.search(r'\\caption\{((?:[^{}]|\{[^{}]*\})*)\}', t, re.S).group(1)
    tabular = re.search(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}', t, re.S)
    spec, rows = tabular.group(1), textmath(tabular.group(2))
    cap = textmath(cap)
    ncol = len(re.sub(r'[^lcrp]', '', spec))
    rows = re.sub(r'\\GROUP\{([^}]*)\}', lambda g: f'\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{g.group(1)}}}}}', rows)
    notes = [textmath(n) for n in re.findall(r'\\NOTE\{((?:[^{}]|\{[^{}]*\})*)\}', t)]
    out = (f'\n\n\\begin{{table}}\n\\caption{{TABLE {ROMAN[tn-1]}. {cap}}}\n'
           f'\\begin{{tabular}}{{{spec}}}{rows}\\end{{tabular}}\n\\end{{table}}\n\n')
    for n in notes:
        out += f'TABLENOTE {n}\n\n'
    return out


body = re.sub(r'\\begin\{table\*?\}(?:\[[^\]]*\])?(.*?)\\end\{table\*?\}', tabfix, body, flags=re.S)

# ---------- 10. abstract, keywords, placeholders ----------
ab = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', body, re.S)
body = body.replace(ab.group(0), '\n\nABSTRACTMARK ' + ' '.join(ab.group(1).split()) + '\n\n')
kw = re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}', body, re.S)
body = body.replace(kw.group(0), '\n\nKEYWORDSMARK ' + ' '.join(kw.group(1).split()) + '\n\n')
body = body.replace('\\maketitle', '')
body = body.replace('\\ALGORITHM', '\n\nALGORITHMMARK\n\n')
body = body.replace('\\REFERENCES', '\n\nREFERENCESMARK\n\n')

title = re.search(r'\\title\{([^}]*)\}', body).group(1)
body = body.replace(re.search(r'\\title\{[^}]*\}', body).group(0), '')
out = ('\\documentclass{article}\n\\title{' + title + '}\n\\begin{document}\n\\maketitle\n'
       + body + '\n\\end{document}\n')
open('pre.tex', 'w', encoding='utf-8').write(out)
json.dump(labels, open('labels.json', 'w'), indent=1)
print(labels)
