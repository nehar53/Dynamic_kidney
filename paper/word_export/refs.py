import re, json
t = open('v3raw.txt', encoding='utf-8').read()
t = t[t.index('R EFERENCES') + len('R EFERENCES'):]
lines = [l.strip() for l in t.split('\n')]
lines = [l for l in lines if l and not re.fullmatch(r'\d{1,2}', l) and '\f' not in l]
entries, cur = [], None
for l in lines:
    l = l.replace('\f', '')
    m = re.match(r'^\[(\d+)\]\s*(.*)', l)
    if m and int(m.group(1)) == len(entries) + 1:
        cur = [m.group(2)]; entries.append(cur)
    elif cur is not None:
        cur.append(l)
out = []
for e in entries:
    s = e[0]
    for l in e[1:]:
        if s.endswith(('/', '%')) or (re.search(r'https?://\S*$', s) and s.endswith('-')):
            s += l
        else:
            s += ' ' + l
    s = re.sub(r'(\d+) (\d{3})', r'\1\2', s)   # 11 496 -> 11496
    s = re.sub(r'\s+', ' ', s).strip()
    out.append(s)
print(len(out))
for i, s in enumerate(out, 1):
    print(i, s)
json.dump(out, open('refs.json', 'w'), ensure_ascii=False, indent=1)
