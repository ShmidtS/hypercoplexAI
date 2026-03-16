"""Сравнительное обучение: Titans vs CLS на real_pairs_v10.json"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import subprocess
import json
import time
from pathlib import Path

REAL_PAIRS = 'data/real_pairs_v10.json'
EPOCHS = 30
COMMON_ARGS = [
    '--real_pairs', REAL_PAIRS,
    '--epochs', str(EPOCHS),
    '--hidden_dim', '64',
    '--pretrained_encoder',
    '--soft_router',
    '--amp',
    '--device', 'cuda',
    '--batch_size', '32',
    '--lr', '3e-4',
    '--lambda_pair', '0.3',
    '--lambda_routing', '0.05',
    '--lambda_memory', '0.01',
    '--use_infonce',
    '--infonce_temperature', '0.15',
    '--eval_every', '5',
    '--save_every', '30',
    '--seed', '42',
    '--train_fraction', '0.85',
    '--scheduler_type', 'cosine_restarts',
]

memory_types = ['titans', 'cls', 'hippocampus', 'neocortex']
results = {}

for mtype in memory_types:
    out_dir = f'artifacts/memory_compare/{mtype}'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, 'scripts/gpu_train.py',
        '--memory_type', mtype,
        '--output_dir', out_dir,
        '--results_json', f'{out_dir}/results.json',
    ] + COMMON_ARGS
    
    print(f'\n{"="*60}')
    print(f'Training memory_type={mtype} ({EPOCHS} epochs)...')
    print(f'{"="*60}')
    sys.stdout.flush()
    
    t0 = time.time()
    proc = subprocess.run(cmd, cwd='E:/hypercoplexAI')
    elapsed = time.time() - t0
    
    # Read results
    results_path = Path(f'{out_dir}/results.json')
    if results_path.exists():
        with open(results_path, encoding='utf-8') as f:
            r = json.load(f)
        quality = r.get('quality', {})
        summary = r.get('training_summary', {})
        results[mtype] = {
            'memory_type': mtype,
            'pair_margin': quality.get('pair_margin', 0),
            'STS_exported': quality.get('STS_exported', 0),
            'score': r.get('score', 0),
            'best_epoch': summary.get('best_epoch', 0),
            'elapsed_s': round(elapsed, 1),
            'returncode': proc.returncode,
        }
    else:
        results[mtype] = {'memory_type': mtype, 'error': 'no results.json', 'returncode': proc.returncode}

# Print comparison table
print('\n' + '='*70)
print(f'{"Memory Type":<15} {"Margin":>10} {"STS":>10} {"Score":>10} {"Best Ep":>8} {"Time(s)":>8}')
print('='*70)
for mtype, r in results.items():
    if 'error' in r:
        print(f'{mtype:<15} ERROR: {r["error"]}')
    else:
        print(
            f'{mtype:<15} '
            f'{r["pair_margin"]:>10.4f} '
            f'{r["STS_exported"]:>10.4f} '
            f'{r["score"]:>10.4f} '
            f'{r["best_epoch"]:>8} '
            f'{r["elapsed_s"]:>8.1f}'
        )
print('='*70)

# Save summary
summary_path = Path('artifacts/memory_compare/summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\nSummary saved to {summary_path}')
