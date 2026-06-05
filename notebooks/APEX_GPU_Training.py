# APEX GPU Training Script for Google Colab
# ============================================================
# HOW TO USE:
# 1. Go to https://colab.research.google.com
# 2. File → New notebook
# 3. Runtime → Change runtime type → T4 GPU
# 4. Copy each section below into a separate cell and run in order
# ============================================================

# ── CELL 1: Verify GPU ──────────────────────────────────────
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')
else:
    raise RuntimeError('No GPU! Go to Runtime → Change runtime type → T4 GPU')
DEVICE = 'cuda'


# ── CELL 2: Install dependencies ────────────────────────────
# !pip install -q torch torchvision scipy pandas numpy faiss-gpu

# ── CELL 3: Download MovieLens-25M ──────────────────────────
import urllib.request, zipfile, os
from pathlib import Path

DATA_DIR = Path('/content/data')
MODELS_DIR = Path('/content/models')
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ML25_DIR = DATA_DIR / 'ml-25m'
if not ML25_DIR.exists():
    print('Downloading MovieLens-25M (250MB)...')
    urllib.request.urlretrieve(
        'https://files.grouplens.org/datasets/movielens/ml-25m.zip',
        DATA_DIR / 'ml-25m.zip'
    )
    with zipfile.ZipFile(DATA_DIR / 'ml-25m.zip', 'r') as z:
        z.extractall(DATA_DIR)
    print('Done.')
else:
    print('Already downloaded.')

import pandas as pd, numpy as np
ratings = pd.read_csv(ML25_DIR / 'ratings.csv')
links = pd.read_csv(ML25_DIR / 'links.csv').dropna(subset=['tmdbId'])
links['tmdbId'] = links['tmdbId'].astype(int)
merged = ratings.merge(links[['movieId','tmdbId']], on='movieId', how='inner')
print(f'Loaded {len(merged):,} ratings, {merged.userId.nunique():,} users, {merged.tmdbId.nunique():,} movies')


# ── CELL 4: Train LightGCN on GPU (200 epochs) ──────────────
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.sparse as sp, time

print('Building LightGCN graph...')
positives = merged[merged['rating'] >= 3.5].copy()
user_ids = sorted(positives['userId'].unique())
item_ids = sorted(positives['tmdbId'].unique())
user_map = {u:i for i,u in enumerate(user_ids)}
item_map = {m:i for i,m in enumerate(item_ids)}
num_users, num_items = len(user_ids), len(item_ids)
print(f'Graph: {num_users:,} users, {num_items:,} items, {len(positives):,} positive interactions')

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

model = LightGCN(num_users, num_items, emb_dim=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

rng = np.random.default_rng(42)
u_all = positives['userId'].map(user_map).values.astype(np.int64)
p_all = positives['tmdbId'].map(item_map).values.astype(np.int64)
BATCH = 8192
EPOCHS = 200

print(f'Training LightGCN for {EPOCHS} epochs on {DEVICE}...')
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    idx = rng.choice(len(u_all), size=min(500000, len(u_all)), replace=False)
    n_arr = rng.integers(0, num_items, size=len(idx)).astype(np.int64)
    perm = rng.permutation(len(idx))
    total_loss, nb = 0.0, 0
    for s in range(0, len(perm), BATCH):
        bi = perm[s:s+BATCH]
        u = torch.tensor(u_all[idx[bi]], dtype=torch.long, device=DEVICE)
        p = torch.tensor(p_all[idx[bi]], dtype=torch.long, device=DEVICE)
        n = torch.tensor(n_arr[bi], dtype=torch.long, device=DEVICE)
        ue = model.user_emb(u); pe = model.item_emb(p); ne = model.item_emb(n)
        loss = F.softplus((ue*ne).sum(1) - (ue*pe).sum(1)).mean()
        loss += 1e-4*(ue.norm(2).pow(2)+pe.norm(2).pow(2)+ne.norm(2).pow(2))/len(bi)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); total_loss += loss.item(); nb += 1
    scheduler.step()
    if (epoch+1) % 20 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/nb:.4f} | {time.time()-t0:.0f}s elapsed')

torch.save({'user_emb': model.user_emb.weight.data.cpu(),
            'item_emb': model.item_emb.weight.data.cpu(),
            'user_ids': user_ids, 'item_ids': item_ids},
           MODELS_DIR / 'lightgcn_ml25m.pth')
print('LightGCN saved.')

# Also save in LightGCN format for APEX
class LightGCNCompat(nn.Module):
    def __init__(self, nu, ni, ed):
        super().__init__()
        self.user_embedding = nn.Embedding(nu, ed)
        self.item_embedding = nn.Embedding(ni, ed)
compat = LightGCNCompat(num_users, num_items, 64)
compat.user_embedding.weight.data = model.user_emb.weight.data.cpu()
compat.item_embedding.weight.data = model.item_emb.weight.data.cpu()
torch.save(compat.state_dict(), MODELS_DIR / 'lightgcn.pth')
print('lightgcn.pth saved (APEX format).')


# ── CELL 5: Export LightGCN embeddings to Gold layer ────────
GOLD_DIR = DATA_DIR / 'gold'
(GOLD_DIR / 'model_user_embeddings').mkdir(parents=True, exist_ok=True)
(GOLD_DIR / 'model_item_embeddings').mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    u_embs = model.user_emb.weight.cpu().numpy()
    i_embs = model.item_emb.weight.cpu().numpy()

pd.DataFrame([{'id': uid, 'features': u_embs[user_map[uid]].tolist()} for uid in user_ids]).to_parquet(
    GOLD_DIR / 'model_user_embeddings' / 'part-0.parquet')
pd.DataFrame([{'id': mid, 'features': i_embs[item_map[mid]].tolist()} for mid in item_ids]).to_parquet(
    GOLD_DIR / 'model_item_embeddings' / 'part-0.parquet')
print(f'Exported embeddings: {len(user_ids):,} users, {len(item_ids):,} items')


# ── CELL 6: Train SASRec on GPU (50 epochs) ─────────────────
print('Building SASRec sequences...')
sorted_r = merged.sort_values(['userId','timestamp'])
item_counts = merged['tmdbId'].value_counts()
popular = set(item_counts[item_counts >= 5].index)
sas_items = sorted(popular)
sas_map = {m:i+1 for i,m in enumerate(sas_items)}
num_sas_items = len(sas_items)

sorted_r2 = sorted_r[sorted_r['tmdbId'].isin(sas_map)].copy()
sorted_r2['item_idx'] = sorted_r2['tmdbId'].map(sas_map)

user_seqs = {}
for uid, grp in sorted_r2.groupby('userId', sort=False):
    seq = grp['item_idx'].tolist()
    if len(seq) >= 3:
        user_seqs[uid] = seq
print(f'SASRec: {len(user_seqs):,} users, {num_sas_items:,} items')

MAX_SEQ = 50

class SASRec(nn.Module):
    def __init__(self, n_items, d=64, n_blocks=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items+1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_SEQ, d)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d*4,
            dropout=dropout, batch_first=True, norm_first=True
        ) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        B, L = x.shape
        e = self.item_emb(x)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B,-1)
        e = self.dropout(e + self.pos_emb(pos))
        mask = torch.triu(torch.ones(L,L,device=x.device,dtype=torch.bool),1)
        for blk in self.blocks:
            e = blk(e, src_mask=mask, is_causal=False)
        return self.norm(e)

sas_model = SASRec(num_sas_items, d=128, n_blocks=3, n_heads=4).to(DEVICE)
sas_opt = torch.optim.Adam(sas_model.parameters(), lr=1e-3)
sas_sched = torch.optim.lr_scheduler.CosineAnnealingLR(sas_opt, T_max=50)

uid_list = list(user_seqs.keys())
SAS_EPOCHS = 50
SAS_BATCH = 512
print(f'Training SASRec for {SAS_EPOCHS} epochs...')
t0 = time.time()
for epoch in range(SAS_EPOCHS):
    sas_model.train()
    sampled = rng.choice(uid_list, size=min(10000, len(uid_list)), replace=False)
    seqs_b, pos_b, neg_b = [], [], []
    for uid in sampled:
        seq = user_seqs[uid]
        i = rng.integers(1, len(seq))
        inp = seq[max(0,i-MAX_SEQ):i]
        inp = [0]*(MAX_SEQ-len(inp)) + inp
        seqs_b.append(inp); pos_b.append(seq[i])
        neg_b.append(int(rng.integers(1, num_sas_items+1)))
    perm = rng.permutation(len(seqs_b))
    total_loss, nb = 0.0, 0
    for s in range(0, len(perm), SAS_BATCH):
        bi = perm[s:s+SAS_BATCH]
        st = torch.tensor([seqs_b[i] for i in bi], dtype=torch.long, device=DEVICE)
        pt = torch.tensor([pos_b[i] for i in bi], dtype=torch.long, device=DEVICE)
        nt = torch.tensor([neg_b[i] for i in bi], dtype=torch.long, device=DEVICE)
        out = sas_model(st)[:,-1,:]
        pe = sas_model.item_emb(pt); ne = sas_model.item_emb(nt)
        loss = F.softplus((out*ne).sum(1)-(out*pe).sum(1)).mean()
        sas_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sas_model.parameters(), 1.0)
        sas_opt.step(); total_loss += loss.item(); nb += 1
    sas_sched.step()
    if (epoch+1) % 10 == 0:
        print(f'  SASRec Epoch {epoch+1}/{SAS_EPOCHS} | Loss: {total_loss/nb:.4f} | {time.time()-t0:.0f}s')

torch.save(sas_model.state_dict(), MODELS_DIR / 'sasrec.pth')
print('sasrec.pth saved.')
