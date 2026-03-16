"""Сравнительное обучение: Titans vs HBMA vs CLS vs Hippocampus с метриками.
Использует реальные текстовые пары из real_pairs_v10.json, энкодит через SBERT,
затем сравнивает 4 типа памяти на одинаковых эмбеддингах."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hdim_model import HDIMConfig, HDIMModel


def load_and_encode(path='data/real_pairs_v10.json'):
    """Загружает текстовые пары, энкодит через SBERT один раз, возвращает (X, D, dim)."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f'  [!] Не удалось загрузить {path}: {e}')
        return _synthetic_data()

    if not isinstance(data, list) or len(data) == 0:
        return _synthetic_data()

    first = data[0]
    if isinstance(first, dict) and 'source_text' in first:
        return _encode_texts(data)
    elif isinstance(first, dict) and 'emb_a' in first:
        dim = len(first['emb_a'])
        X = torch.tensor([d['emb_a'] for d in data] + [d['emb_b'] for d in data], dtype=torch.float32)
        D = torch.zeros(len(X), dtype=torch.long)
        return X, D, dim
    else:
        return _synthetic_data()


def _synthetic_data():
    print('  [!] Используем синтетические данные')
    dim = 768
    X = torch.randn(2048, dim)
    D = torch.randint(0, 4, (2048,))
    return X, D, dim


def _encode_texts(data):
    """Энкодит текстовые пары через SBERT (один раз, без градиентов)."""
    print('  [SBERT] Загрузка энкодера...')
    from sentence_transformers import SentenceTransformer

    device = 'cpu'  # Энкодинг на CPU чтобы не мешать training на GPU
    encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    encoder.to(device)

    # Собираем все уникальные тексты
    texts = set()
    for item in data:
        texts.add(item['source_text'])
        texts.add(item['target_text'])
    texts = list(texts)
    print(f'  [SBERT] Кодируем {len(texts)} уникальных текстов на CPU...')

    with torch.no_grad():
        embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_tensor=True, device=device)

    text_to_emb = {t: embeddings[i].cpu() for i, t in enumerate(texts)}

    # Строим пары: source → target
    X_list = []
    D_list = []
    for item in data:
        src_emb = text_to_emb[item['source_text']]
        tgt_emb = text_to_emb[item['target_text']]
        X_list.append(src_emb)
        X_list.append(tgt_emb)
        src_dom = item.get('source_domain', 0)
        tgt_dom = item.get('target_domain', 0)
        D_list.append(src_dom)
        D_list.append(tgt_dom)

    X = torch.stack(X_list).float()
    D = torch.tensor(D_list, dtype=torch.long)
    dim = X.shape[1]
    print(f'  [SBERT] Готово: {len(X)} эмбеддингов, dim={dim}')
    return X, D, dim


def train_one_memory(
    memory_type: str,
    X, D, dim,
    hidden_dim: int = 256,
    num_domains: int = 4,
    num_experts: int = 4,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = 'cpu',
    seed: int = 42,
):
    """Обучает HDIMModel с заданным memory_type и возвращает метрики."""
    torch.manual_seed(seed)

    # Проекция до hidden_dim если нужно
    if dim != hidden_dim:
        proj = nn.Linear(dim, hidden_dim)
        X = proj(X)

    # Detach чтобы не было computation graph bleed
    X = X.detach().clone()
    D = D.detach().clone()

    dataset = torch.utils.data.TensorDataset(X, D)
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Модель
    cfg = HDIMConfig(
        hidden_dim=hidden_dim,
        num_domains=num_domains,
        num_experts=num_experts,
        memory_type=memory_type,
    )
    model = HDIMModel(cfg).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_b, D_b in train_loader:
            X_b, D_b = X_b.to(device), D_b.to(device)
            optimizer.zero_grad()
            out, rw, inv = model(X_b, D_b)
            loss = nn.functional.mse_loss(out, X_b)

            # Routing entropy regularization
            eps = 1e-8
            rw_norm = rw / (rw.sum(dim=-1, keepdim=True) + eps)
            entropy = -(rw_norm * (rw_norm + eps).log()).sum(dim=-1).mean()
            loss = loss - 0.01 * entropy

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

    elapsed = time.time() - t0

    # Валидация
    model.eval()
    val_loss = 0.0
    n_val_batches = 0
    with torch.no_grad():
        for X_b, D_b in val_loader:
            X_b, D_b = X_b.to(device), D_b.to(device)
            out, rw, inv = model(X_b, D_b)
            loss = nn.functional.mse_loss(out, X_b)
            val_loss += loss.item()
            n_val_batches += 1
    val_loss /= max(n_val_batches, 1)

    return {
        'memory_type': memory_type,
        'n_params': n_params,
        'train_loss_epoch1': round(train_losses[0], 4),
        'train_loss_final': round(train_losses[-1], 4),
        'val_loss': round(val_loss, 4),
        'time_s': round(elapsed, 1),
        'epochs': epochs,
        'improvement': round(train_losses[0] - train_losses[-1], 4),
    }


def main():
    print('=' * 72)
    print('HDIM Memory Comparison — Titans vs HBMA vs CLS vs Hippocampus')
    print('=' * 72)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print()

    # Загружаем и энкодим данные ОДИН раз
    print('Loading and encoding data...')
    X, D, dim = load_and_encode()
    print(f'Using dim={dim}, samples={len(X)}')
    print()

    memory_types = ['titans', 'hbma', 'cls', 'hippocampus']
    results = {}

    for mtype in memory_types:
        print(f'Training {mtype} (20 epochs)...', end=' ', flush=True)
        try:
            r = train_one_memory(mtype, X, D, dim, epochs=20, device=device)
            results[mtype] = r
            print(f'done ({r["time_s"]}s)')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'ERROR: {e}')
            results[mtype] = {'memory_type': mtype, 'error': str(e)}

    # Таблица
    print()
    print('=' * 72)
    print(f'{"Memory Type":<15} {"Params":>8} {"Ep1 Loss":>10} {"Final":>10} {"Val":>10} {"Δ":>8} {"Time":>8}')
    print('=' * 72)
    for mtype in memory_types:
        r = results[mtype]
        if 'error' in r:
            print(f'{mtype:<15} ERROR: {r["error"][:60]}')
        else:
            print(
                f'{r["memory_type"]:<15} '
                f'{r["n_params"]:>8,} '
                f'{r["train_loss_epoch1"]:>10.4f} '
                f'{r["train_loss_final"]:>10.4f} '
                f'{r["val_loss"]:>10.4f} '
                f'{r["improvement"]:>8.4f} '
                f'{r["time_s"]:>7.1f}s'
            )
    print('=' * 72)

    # Сохранение
    out = Path('artifacts/memory_train_comparison.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
