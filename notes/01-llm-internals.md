# 01 — Modern LLM internals (the parts that matter for token reduction)

## Why this matters

Visual-token reduction exists almost entirely because **LLM inference cost is dominated by the KV cache and quadratic attention**. If you don't have a crisp picture of how decoding works, RoPE rotates positions, and FlashAttention reshapes the memory bottleneck, every reduction paper will read like noise. After this module you should be able to point at the picture and say "the win comes *here*."

## Prerequisites

You already know transformers and attention. We're only filling in: causal/decoder-only specifics, the inference-time KV cache, RoPE, and a one-paragraph picture of FlashAttention.

## Learning path

### 1. Decoder-only causal language modeling

What it is: same transformer block you know, but with a *causal* mask so position *t* only attends to positions ≤ *t*. At inference, you generate one token at a time. The same block is used both for "prefill" (encoding the prompt all at once) and "decode" (one new token at a time).

- **Primary:** [The Illustrated GPT-2 (Jay Alammar)](https://jalammar.github.io/illustrated-gpt2/) — visual, ~15-min read. Skip the parts you already know; pay attention to the decoder-block diagrams and the "one token at a time" inference picture.
- **Go deeper:** [Karpathy — Let's build GPT (1h56m)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — overkill for this study but the single best ground-truth for what's actually going on. Watch the chunks on causal masking and the generation loop if Alammar didn't land.

### 2. KV cache — the central object

What it is: during generation, at each new step you only need the **new** query; the keys and values for all previous tokens never change. So you cache them and only do O(T) attention per step instead of recomputing O(T²) every time. Inference memory grows linearly with sequence length — and visual tokens are *long sequences*. This is why a 1-minute video at 8 fps can OOM a 24-GB GPU.

- **Primary video:** [KV Cache in LLMs Explained Visually — YouTube](https://www.youtube.com/watch?v=7OrMFn86PlM)
- **Go deeper (blog):** [Sebastian Raschka — Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)

### 3. RoPE (Rotary Position Embedding)

What it is: instead of *adding* a position embedding to each token, RoPE *rotates* `q` and `k` by an angle that depends on position. Dot-products then depend only on the *relative* position. Critical to know because (a) many token-pruning methods worry about whether their reduction preserves the position information RoPE encodes, and (b) video VLMs often use M-RoPE / 3D-RoPE to encode (time, height, width) jointly.

- **Primary:** [EleutherAI — Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) — short, with the geometric picture.
- **Go deeper:** Reading-list relevance — when you hit a paper that says "we keep RoPE indices fixed while merging tokens", come back here.

### 4. FlashAttention (one-paragraph picture)

The original quadratic attention is *compute*-bound on paper, but on real GPUs it's *memory-IO*-bound — the bottleneck is shuttling the big T×T attention matrix in and out of HBM. FlashAttention tiles the attention computation so the T×T matrix never materializes in HBM; everything happens in fast on-chip SRAM. Same math, exact, just IO-aware. The takeaway for token-reduction work: **dropping tokens directly reduces the FlashAttention tile count and so directly reduces wall-clock time, which is why so many papers measure tokens/sec rather than only FLOPs.**

- **Primary:** [DigitalOcean — Designing Hardware-Aware Algorithms: FlashAttention](https://www.digitalocean.com/community/tutorials/flashattention) — short tutorial.
- **Go deeper:** Dao et al., *FlashAttention*, NeurIPS 2022 — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

## Code exercise — a minimal KV cache

Run this in any environment with `torch`. ~25 lines. The goal is to internalize "the cache appends one row per step; the new query attends to the whole cache."

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
D = 16
Wq, Wk, Wv = (torch.randn(D, D) for _ in range(3))

def attend(q, K, V):                                # q: (1, D); K, V: (T, D)
    s = (q @ K.T) / D**0.5                          # (1, T)
    return F.softmax(s, dim=-1) @ V                 # (1, D)

# Without cache: recompute K, V for the whole prefix every step.
def step_no_cache(tokens):                          # tokens: (T, D)
    K, V = tokens @ Wk, tokens @ Wv                 # O(T * D^2) each call
    return attend(tokens[-1:] @ Wq, K, V)

# With cache: append the new token's K, V; query attends to all of them.
class KVCache:
    def __init__(self):
        self.K = self.V = None
    def step(self, tok):                            # tok: (1, D)
        k, v = tok @ Wk, tok @ Wv                   # O(D^2) only
        self.K = k if self.K is None else torch.cat([self.K, k], 0)
        self.V = v if self.V is None else torch.cat([self.V, v], 0)
        return attend(tok @ Wq, self.K, self.V)

seq = torch.randn(5, D)
cache = KVCache()
for t in seq:                                       # generate "token by token"
    cache.step(t.unsqueeze(0))

print("no-cache out norm:", step_no_cache(seq).norm().item())
print("cached   out norm:", attend(seq[-1:] @ Wq, cache.K, cache.V).norm().item())
print("cache K shape:", tuple(cache.K.shape))      # (5, 16) — grew per step
```

Two things to notice once it runs:
- The two output norms are equal (within float error) — the cache is just a *bookkeeping* optimization, not a different algorithm.
- The cache `K` shape grows with the sequence. For a VLM, that growth is dominated by visual tokens. Token reduction = directly attacking this object.

## Self-check

- Explain to your manager, in two sentences, **why a VLM's visual token count is more painful at inference than at training.** (Training pays it once; decode pays it on every generated text token through the KV cache.)
- If you halve the number of visual tokens before they enter the LLM, **what speedup do you expect on prefill? On per-token decode?** (Prefill ~4×: attention is quadratic. Decode ~2×: attention there is linear in cache length.)
- **Where in the pipeline does RoPE actually get applied?** (Inside attention, on `q` and `k`, *before* the dot product. Not on the residual stream.)
- **Why is FlashAttention exact and not approximate?** (It tiles the same softmax computation, never materializing the full T×T matrix in HBM; the math is identical.)
- **Mechanism question:** if you randomly drop 50% of visual tokens *after* the LLM has already filled its KV cache, what does the saved compute look like for the *next* generated token vs. the *next 100* generated tokens? (Per-token: linear savings; cumulative: linear × tokens generated. This is the decode-side win you'll see in DyCoke etc.)

## My notes

(fill in as you go)

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — arXiv:1706.03762.
- Su et al., *RoFormer (RoPE)*, 2021 — arXiv:2104.09864.
- Dao et al., *FlashAttention*, NeurIPS 2022 — arXiv:2205.14135.
- Dao, *FlashAttention-2*, 2023 — arXiv:2307.08691.
