"""Static, standard-library-only audit. Does not invoke a TeX compiler."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
errors = []
checks = []

def require(condition, description):
    checks.append(description)
    if not condition:
        errors.append(description)

def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)

def argument(text, start):
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != '{':
        raise ValueError('Expected brace argument: ' + text[start:start + 80])
    depth = 1
    end = start + 1
    while depth:
        if text[end] == '{' and text[end - 1] != '\\':
            depth += 1
        if text[end] == '}' and text[end - 1] != '\\':
            depth -= 1
        end += 1
    return text[start + 1:end - 1], end

tex_files = sorted(ROOT.rglob('*.tex'))
text = '\n'.join(p.read_text(encoding='utf-8') for p in tex_files)
require((ROOT / 'main.tex').exists(), 'Root main.tex exists')
inputs = []
graphics = []
for p in tex_files:
    s = uncomment(p.read_text(encoding='utf-8'))
    for target in re.findall(r'\\(?:input|include)\{([^}]+)\}', s):
        target = target if target.endswith('.tex') else target + '.tex'
        dest = (ROOT / target).resolve()
        require(dest.is_relative_to(ROOT) and dest.is_file(), 'Input resolves inside report: ' + target)
        inputs.append(dest)
    for target in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', s):
        dest = (ROOT / target).resolve()
        require(dest.is_relative_to(ROOT) and dest.is_file(), 'Figure resolves inside report: ' + target)
        graphics.append(target)
    # Environments must nest correctly in each source file.
    stack = []
    for kind, env in re.findall(r'\\(begin|end)\{([^}]+)\}', s):
        if kind == 'begin':
            stack.append(env)
        else:
            require(bool(stack) and stack[-1] == env, 'Environment closure: ' + p.name + ':' + env)
            if stack:
                stack.pop()
    require(not stack, 'All environments closed: ' + p.name)
    syntax = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', '', s, flags=re.S)
    depth = 0
    for match in re.finditer(r'(?<!\\)[{}]', syntax):
        depth += 1 if match.group() == '{' else -1
        if depth < 0:
            break
    require(depth == 0, 'Balanced braces: ' + p.name)
    require(len(re.findall(r'(?<!\\)\$', syntax)) % 2 == 0, 'Paired math dollars: ' + p.name)
    require(not re.search(r'\bTODO\b|\bTBD\b|insert result here|placeholder', s, re.I), 'No placeholder text: ' + p.name)
    for match in re.finditer(r'\\begin\{(tabular|tabularx|longtable)\}', s):
        env = match.group(1)
        spec, end = argument(s, match.end())
        if env == 'tabularx':
            spec, end = argument(s, end)
        # Count only top-level column specifiers, ignoring width and decorator arguments.
        cols = 0
        i = 0
        while i < len(spec):
            if spec[i] == '{':
                _, i = argument(spec, i)
                continue
            if spec[i] in 'lcrXYpmb':
                cols += 1
            i += 1
        body = s[end:s.index('\\end{' + env + '}', end)]
        for row in re.split(r'\\\\(?:\[[^\]]*\])?', body):
            if '&' not in row and '\\multicolumn' not in row:
                continue
            cells = re.split(r'(?<!\\)&', row)
            actual = sum(int(m.group(1)) if (m := re.search(r'\\multicolumn\{(\d+)\}', cell)) else 1 for cell in cells)
            require(actual == cols, f'Table row has {cols} columns: {p.name}: ' + row.strip()[:70])

labels = re.findall(r'\\label\{([^}]+)\}', text)
require(not [k for k, n in Counter(labels).items() if n > 1], 'No duplicate labels')
for targets in re.findall(r'\\(?:[Cc]ref|ref|eqref|pageref)\{([^}]+)\}', text):
    for target in targets.split(','):
        require(target.strip() in labels, 'Reference has label: ' + target.strip())
bib = (ROOT / 'references.bib').read_text(encoding='utf-8')
bibkeys = re.findall(r'@\w+\s*\{([^,\s]+)', bib)
require(len(bibkeys) == len(set(bibkeys)), 'No duplicate bibliography keys')
citations = set()
for targets in re.findall(r'\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}', text):
    for target in targets.split(','):
        citations.add(target.strip())
        require(target.strip() in bibkeys, 'Citation has bibliography entry: ' + target.strip())
require(set(bibkeys) == citations, 'Every bibliography entry is cited')
require(set(tex_files) - {ROOT / 'main.tex'} <= set(inputs), 'All section source files are included')
for target in re.findall(r'\\bibliography\{([^}]+)\}', text):
    require((ROOT / (target + '.bib')).is_file(), 'Bibliography file exists')
for token in ['yuxuanliang1', 'MLDS490-Assignment3', 'https://github.com/yuxuanliang1/MLDS490-Assignment3']:
    require(token in (ROOT / 'main.tex').read_text(), 'First-page repository identifier: ' + token)
require(len(re.findall(r'\\subsection\{Scientific Validation and Edge Cases\}', text)) == 3,
        'All three parts have dedicated validation subsections')
require(all(x in text for x in ['Month 1:', 'Month 2:', 'Month 3:']), 'All three future-work phases present')
for item in json.loads((ROOT / 'FIGURE_MANIFEST.json').read_text()):
    require(hashlib.sha256((ROOT / item['report_path']).read_bytes()).hexdigest() == item['sha256'],
            'Figure integrity: ' + item['report_path'])
require(set(graphics) == {p.relative_to(ROOT).as_posix() for p in (ROOT / 'figures').rglob('*') if p.suffix in ['.pdf', '.png']},
        'Every packaged figure is used; every used figure is packaged')
for phrase in ['report preparation', 'source audit', 'SOURCE_AUDIT', 'SOURCE\\_AUDIT',
               'copied source PDF', 'stale status', 'no upload or push',
               'earlier exploratory Windows', 'validation-loss allowance', '/nfs/home/', 'Assignment2']:
    require(phrase.lower() not in text.lower(), 'No inappropriate narrative: ' + phrase)
for label in ['tab:validation_summary', 'eq:selection', 'fig:benefit', 'fig:timeline', 'fig:robustness', 'app:reproduction']:
    require(label in labels, 'Required synthesis or evidence label: ' + label)
require('Limitations and Scope of Interpretation' in text, 'Consolidated limitations present')
require('maximum two-percentage-point decrease in mean final validation accuracy' in text,
        'Selection criterion refers to validation accuracy')
require(not [p for p in ROOT.rglob('*') if p.suffix in ['.npy', '.npz', '.pt', '.pth']], 'No raw data or model checkpoints in package')
require(not list(ROOT.glob('*.pdf')), 'No compiled report PDF')
result = dict(passed=not errors, check_count=len(checks), errors=errors,
              tex_files=len(tex_files), unique_figure_assets=len(set(graphics)),
              numbered_figure_environments=len(re.findall(r'\\begin\{figure\}', text)),
              subfigures=len(re.findall(r'\\begin\{subfigure\}', text)),
              numbered_tables=len(re.findall(r'\\begin\{(?:table|longtable)\}', text)),
              bibliography_entries=len(bibkeys), labels=len(labels),
              compilation_performed=False,
              limitation='Static source checks do not establish successful compilation, page count, float placement, or absence of overfull boxes.')
(ROOT / 'STATIC_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
raise SystemExit(1 if errors else 0)
