#!/usr/bin/env python
"""Слияние датасетов real_pairs_v2.json + real_pairs_v3_extension.json → real_pairs_v4.json"""
import json
from pathlib import Path

base = Path('E:/hypercoplexAI/data')

v2 = json.loads((base / 'real_pairs_v2.json').read_text())
ext = json.loads((base / 'real_pairs_v3_extension.json').read_text())

# Пересчитать group_id в расширении чтобы не конфликтовали
max_gid = max(p['group_id'] for p in v2)
for p in ext:
    if p['group_id'] < 200:  # уже OK (200+)
        p['group_id'] += max_gid + 1

merged = v2 + ext

# Статистика
pos = [p for p in merged if p['relation'] == 'positive']
neg = [p for p in merged if p['relation'] == 'negative']
domains = {}
for p in merged:
    for d in [p['source_domain'], p['target_domain']]:
        domains[d] = domains.get(d, 0) + 1

print(f'Merged: {len(merged)} pairs ({len(pos)} pos + {len(neg)} neg)')
print(f'Domain distribution: {domains}')
print(f'Unique families: {len(set(p["family"] for p in merged))}')

out = base / 'real_pairs_v4.json'
out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
print(f'Saved to: {out}')
