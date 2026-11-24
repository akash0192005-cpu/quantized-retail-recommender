"""Synthetic basket generation, next-item learning and post-training quantization."""
from __future__ import annotations
import copy, io, random
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CATALOG = ["bread","milk","eggs","butter","cheese","yogurt","apples","bananas","oranges","chicken","rice","pasta","tomatoes","lettuce","cereal","coffee","tea","cookies","orange_juice","potatoes"]
THEMES = [
 ["bread","milk","eggs","butter"], ["pasta","tomatoes","cheese"],
 ["chicken","rice","lettuce","tomatoes"], ["cereal","milk","bananas"],
 ["coffee","cookies","milk"], ["yogurt","apples","bananas"],
 ["bread","cheese","lettuce","tomatoes"], ["potatoes","chicken","butter"],
]

class RecommenderNet(nn.Module):
    """Compact multi-hot basket-context to next-item predictor."""
    def __init__(self, n_items: int, hidden: int = 64):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.fc1 = nn.Linear(n_items, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, n_items)
        self.dequant = torch.ao.quantization.DeQuantStub()
    def forward(self, x):
        return self.dequant(self.fc2(self.relu(self.fc1(self.quant(x)))))

@dataclass
class TrainingResult:
    model: nn.Module
    items: list[str]
    train_loss: float
    accuracy: float
    examples: int

def generate_baskets(n: int = 1600, seed: int = 42) -> list[list[str]]:
    """Generate deterministic synthetic transactions with co-purchase patterns."""
    if n < 1: raise ValueError("n must be at least 1")
    rng = random.Random(seed); baskets = []
    for _ in range(n):
        theme = list(rng.choice(THEMES)); basket = set(theme)
        for item in CATALOG:
            if item not in basket and rng.random() < .025: basket.add(item)
        extras = [x for x in basket if x not in theme]; rng.shuffle(extras)
        rng.shuffle(theme); baskets.append(theme + extras)
    return baskets

def _examples(baskets, items):
    idx = {v:i for i,v in enumerate(items)}; xs=[]; ys=[]
    for basket in baskets:
        ordered=[x for x in basket if x in idx]
        for pos,item in enumerate(ordered):
            row=np.zeros(len(items),dtype=np.float32)
            for previous in ordered[:pos]: row[idx[previous]]=1.
            xs.append(row); ys.append(idx[item])
    if not xs: raise ValueError("No training examples")
    return torch.tensor(np.asarray(xs)), torch.tensor(ys,dtype=torch.long)

def _accuracy(model,x,y):
    model.eval()
    with torch.no_grad(): pred=model(x).argmax(1)
    return float((pred==y).float().mean().item())

def train_recommender(baskets=None, epochs=18, hidden=64, seed=42):
    """Train a small next-item model on basket prefixes with a basket-level holdout."""
    torch.manual_seed(seed); np.random.seed(seed)
    baskets = baskets or generate_baskets(seed=seed)
    items=sorted({x for b in baskets for x in b})
    shuffled=list(baskets); random.Random(seed).shuffle(shuffled)
    split=max(1,int(len(shuffled)*.8))
    tx,ty=_examples(shuffled[:split],items)
    vx,vy=_examples(shuffled[split:] or shuffled[:1],items)
    model=RecommenderNet(len(items),hidden)
    opt=torch.optim.Adam(model.parameters(),lr=.01); loss_fn=nn.CrossEntropyLoss()
    loader=DataLoader(TensorDataset(tx,ty),batch_size=128,shuffle=True)
    loss_value=0.
    for _ in range(max(1,epochs)):
        model.train(); total=count=0
        for xb,yb in loader:
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
            total += float(loss.item())*len(yb); count += len(yb)
        loss_value=total/max(1,count)
    return TrainingResult(model,items,loss_value,_accuracy(model,vx,vy),len(ty))

def recommend_next(model, history, items, top_k=5):
    """Return top-k suggestions and scores, excluding products already in the basket."""
    if not items: return []
    idx={v:i for i,v in enumerate(items)}
    x=torch.zeros((1,len(items)),dtype=torch.float32)
    for item in history:
        if item in idx: x[0,idx[item]]=1.
    model.eval()
    with torch.no_grad(): scores=model(x)[0].clone()
    for item in history:
        if item in idx: scores[idx[item]]=-torch.inf
    ids=torch.topk(scores,min(max(1,top_k),len(items))).indices.tolist()
    return [(items[i],float(scores[i].item())) for i in ids if torch.isfinite(scores[i])]

def probe_itemsets(baskets,min_support=.05):
    """Rank co-purchased item pairs by transaction support."""
    if not baskets: return []
    counts={}
    for basket in baskets:
        unique=sorted(set(basket))
        for i,left in enumerate(unique):
            for right in unique[i+1:]:
                pair=(left,right); counts[pair]=counts.get(pair,0)+1
    total=len(baskets)
    return [{"items":pair,"support":count/total} for pair,count in
            sorted(counts.items(),key=lambda kv:kv[1],reverse=True) if count/total>=min_support]

def model_size_bytes(model):
    buffer=io.BytesIO(); torch.save(model.state_dict(),buffer); return buffer.tell()

def quantize_dynamic(model):
    """Post-training dynamic INT8 quantization for Linear layers."""
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).eval(),{nn.Linear},dtype=torch.qint8)

def make_calibration_data(baskets,items):
    x,_=_examples(baskets,items); return x

def quantize_static(model,calibration_x):
    """Calibrate and convert a static INT8 model using representative basket contexts."""
    if calibration_x.numel()==0: raise ValueError("Calibration data cannot be empty")
    engine="x86" if "x86" in torch.backends.quantized.supported_engines else "qnnpack"
    torch.backends.quantized.engine=engine
    prepared=copy.deepcopy(model).cpu().eval()
    prepared.qconfig=torch.ao.quantization.get_default_qconfig(engine)
    prepared=torch.ao.quantization.prepare(prepared,inplace=False)
    with torch.no_grad():
        for batch in calibration_x.cpu().split(128): prepared(batch)
    return torch.ao.quantization.convert(prepared,inplace=False)

def evaluate_model(model,baskets,items):
    x,y=_examples(baskets,items); return _accuracy(model,x,y)
