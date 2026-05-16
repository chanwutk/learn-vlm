# 06 — Visual token-reduction toolbox

## Why this matters

This is the heart of the internship topic. Every paper in `SUGGESTED_READINGS.md`'s "Token Compression & Reduction" and "Adaptive Frame & Spatial Selection" sections is built on the ideas in this module. After this you should be able to (a) sketch FastV and ToMe on a whiteboard from memory, (b) place any new paper into the taxonomy, and (c) start brainstorming combinations and weaknesses.

## Prerequisites

Modules 01 (KV cache), 03 (where the tokens live), 05 (the video token-count math). The "taxonomy" picture from Module 00 is the spine of this module.

## Learning path

### 1. The taxonomy, re-stated with concrete methods

| Bucket | Canonical method | One-line idea | Trained? | Where it acts |
|---|---|---|---|---|
| **Attention-score pruning** | **FastV** | Keep tokens the last text token attends to most; drop the rest after a shallow LLM layer. | training-free | inside LLM (after layer 2) |
| **Similarity merging** | **ToMe** | Pair up similar tokens, average each pair into one. | training-free | inside encoder or LLM |
| **Encoder attention + merge** | **VisionZip** | Score by encoder self-attention; keep top-K dominant tokens, merge the rest by similarity. | training-free | between encoder and LLM |
| **Query-based aggregation** | **Q-Former** | Fixed N learnable queries cross-attend to many visual tokens. | trained | connector |
| **KV-cache eviction** | **H2O** | At decode time, keep "heavy-hitter" tokens (those that got high attention so far) and recent tokens; evict the rest. | training-free | LLM decode loop |
| **KV-cache streaming** | **StreamingLLM** | Keep a few initial "attention sinks" + the recent window; evict the middle. | training-free | LLM decode loop |
| **Adaptive frame sampling** | **VideoBrain** (Module 07) | Learn to pick the right frames before they're encoded. | trained | pre-encoder |

The trick to discussing the field is that **most papers are *hybrids***: VisionZip is "select then merge," DyCoke is "merge in prefill, evict from KV in decode," LLaVA-UHD is "slice + compress." When reading a paper, build the path: **where in the pipeline** → **select or merge or evict** → **training-free or trained**.

### 2. FastV — read this twice

The whole paper distills to one observation: in trained VLMs, the attention from the **last text token** to **visual tokens** decays sharply after the first few LLM layers. After ~layer 2, most visual tokens are nearly ignored. So: **after layer 2, drop the R% lowest-attended visual tokens.** This is the simplest training-free reducer that actually works — 45% FLOPs reduction at no accuracy cost.

What to remember:
- Importance metric: attention from the *last* text-side token (the "instruction-end" or `[CLS]`-like pivot).
- Where to prune: after a shallow LLM layer, not at the encoder. The encoder's tokens haven't "decided" what's important yet; the LLM has.
- Why it generalizes: this isn't a tuned hyperparameter — the layer-2 cutoff is consistent across most LLaVA-family models.

- **Primary:** [FastV GitHub README (pkunlp-icler)](https://github.com/pkunlp-icler/FastV) — figures + a 1-paragraph explanation.
- **Go deeper:** Chen et al., *An Image is Worth 1/2 Tokens After Layer 2 (FastV)*, ECCV 2024 Oral — [arXiv:2403.06764](https://arxiv.org/abs/2403.06764). Read §1 (intro), §3 (method), and Figure 2 (the layer-by-layer attention plot — that's the whole insight).

### 3. ToMe — read this twice too

Whole paper in one sentence: **gradually merge similar tokens by bipartite soft matching, no retraining needed.** Algorithm at one transformer block:
1. Split tokens into sets A (even indices) and B (odd indices).
2. Compute cosine similarity between each `a ∈ A` and each `b ∈ B` using their K vectors.
3. For each `a`, pick its best match in B.
4. Pick the top-r edges (highest similarity); merge those A tokens into their B match (weighted average).
5. Return: surviving A tokens + (possibly averaged) B tokens. Total count drops by r.

Why it works without retraining: the merge is on K (keys), which already encode token similarity in a way attention uses; merging similar tokens barely changes downstream attention. **2× ViT throughput at 0.2-0.3% accuracy loss.**

Video extensions (TempMe, HoliTom, DyCoke's merge step) all build on this template — they typically merge across both space *and* time.

- **Primary:** [Token Merging visual explainer (Medium)](https://medium.com/superlinear-eu-blog/representation-learning-breakthroughs-token-merging-your-vit-but-faster-e3f88f25d6d1) — has the bipartite-matching figure.
- **Go deeper:** Bolya et al., *Token Merging: Your ViT But Faster*, ICLR 2023 Oral — [arXiv:2210.09461](https://arxiv.org/abs/2210.09461). §3 (the algorithm) is 1.5 pages — read it carefully.

### 4. VisionZip — selection + merging combined

What it does: at the connector boundary, **select** dominant tokens using the vision encoder's own self-attention scores (training-free), then **merge** the rest by semantic similarity. End up with ~10% of the original visual tokens with most of the accuracy retained.

The interesting structural point: VisionZip is purely *encoder-side* — it makes a decision before the LLM is involved. This is the opposite philosophy of FastV (which uses LLM attention). Comparing the two side-by-side is a great conversation to have with your manager:
- **FastV:** "Trust the LLM's attention — let it tell us what matters once it sees the question."
- **VisionZip:** "The encoder already knows what's salient; LLM attention is downstream and slower to compute."

- **Primary:** [VisionZip GitHub repo (project page)](https://github.com/dvlab-research/VisionZip) — README with figures.
- **Go deeper:** Yang et al., *VisionZip*, CVPR 2025 — [arXiv:2412.04467](https://arxiv.org/abs/2412.04467).

### 5. KV-cache compression (H2O, StreamingLLM)

These are LLM-side techniques developed for text-only models, but they apply directly to VLMs at decode time.
- **H2O — Heavy-Hitter Oracle:** At each decode step, keep the tokens that have accumulated the most attention so far ("heavy hitters") plus a recency window. The accumulated-attention distribution is a power law, so a small set dominates. Result: keep ~20% of the cache with minimal loss.
- **StreamingLLM:** Notice that the very first few tokens act as "attention sinks" — the model dumps probability mass on them. Keep those + the recent window; evict the middle. Lets you decode arbitrarily long with bounded cache.

For VLMs, these methods kick in *during decoding* — they're complementary to encoder-side reduction. DyCoke explicitly combines a ToMe-style prefill merge with KV-cache eviction at decode (Module 07).

- **Primary:** [Top 10 KV Cache Compression Techniques (MarkTechPost)](https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques-for-llm-inference-reducing-memory-overhead-across-eviction-quantization-and-low-rank-methods/) — short survey, gives you the landscape.
- **Go deeper:** Zhang et al., *H2O*, NeurIPS 2023 — [arXiv:2306.14048](https://arxiv.org/abs/2306.14048). Xiao et al., *StreamingLLM*, ICLR 2024 — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).

### 6. Training-free vs trained — when to use which

| Property | Training-free | Trained |
|---|---|---|
| Cost to deploy | Plug in. | Need a fine-tune pass. |
| Coupling | Loosely coupled — works on any LLaVA-likeish model. | Tightly coupled to the host model. |
| Performance ceiling | Limited by how lucky the pretrained model's attention pattern is. | Higher, especially at extreme reduction (≤10% of tokens). |
| Reading-list examples | FastV, ToMe, DyCoke, "Less is More", VisionZip, ESV | VideoBrain, VideoINSTA, STORM, LLaVA-UHD's compressor |

A good research seed for goal #2: **a training-free policy that adapts at inference based on the question** (text-conditioned but no fine-tune). The reading list has hints of this already; combining ideas across cells of this table is where novelty lives.

## Code exercise — ToMe merge + FastV prune in ~30 lines

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
N, D = 100, 32                              # 100 tokens, 32-dim each
tokens = torch.randn(N, D)
last_q = torch.randn(1, D)                  # imagine: last text token's query

# --- ToMe: bipartite soft-matching merge (training-free) ---
def tome_merge(x, r):                       # cut token count by r
    a, b = x[0::2], x[1::2]                 # bipartite split
    sim = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T   # (|A|,|B|)
    edge_score, edge_idx = sim.max(dim=-1)  # best B match for each A
    order = edge_score.argsort(descending=True)
    merge, keep = order[:r], order[r:]      # merge top-r edges; keep the rest
    merged_b = b.clone()
    merged_b.index_add_(0, edge_idx[merge], a[merge])           # sum a into matched b
    counts = torch.ones(b.size(0))
    counts.index_add_(0, edge_idx[merge], torch.ones(r))        # how many merged in
    return torch.cat([a[keep], merged_b / counts[:, None]], 0)  # avg + concat

# --- FastV: keep tokens the last query attends to most ---
def fastv_prune(x, q, keep_frac):
    s = F.softmax((q @ x.T) / x.size(-1) ** 0.5, dim=-1).squeeze(0)   # (N,)
    k = int(keep_frac * x.size(0))
    return x[s.topk(k).indices]

merged = tome_merge(tokens, r=20)               # 100 → 80 tokens
pruned = fastv_prune(tokens, last_q, 0.5)       # 100 → 50 tokens
print("orig  :", tuple(tokens.shape))
print("ToMe  :", tuple(merged.shape),   "  (merged 20 of 50 pair slots)")
print("FastV :", tuple(pruned.shape),   "  (kept top 50% by attn to last query)")
```

After running:
- Notice **ToMe is text-blind** — it doesn't see the question. It removes redundancy *within* the visual tokens.
- Notice **FastV is text-aware** — it asks "what does the LLM care about given this question?"
- These two are *complementary*, not competing. A realistic system uses both.

## Self-check

- **State FastV's whole method in 2 sentences.** (Run the first ~2 LLM layers normally. After layer 2, drop the visual tokens with the lowest attention from the last text token.)
- **State ToMe's whole method in 2 sentences.** (Bipartite-split tokens into A/B. Find each A's most-similar B, merge the top-r pairs by averaging.)
- **VisionZip uses encoder self-attention; FastV uses LLM attention to the last text token. Why might one be better for OCR-heavy tasks, the other for general VQA?** (Encoder attention is content-driven and language-agnostic — good when the question doesn't pre-select regions. LLM attention is question-driven — good when the prompt tells you what to look at.)
- **Why does H2O preserve power-law "heavy hitters" rather than just the most recent tokens?** (Because attention is not local — important context can be far back in the sequence. Recency-only would drop it.)
- **Mechanism question (open):** suppose you apply ToMe to compress a video to 50% tokens, then apply H2O at decode time to keep 30% of the KV cache. Do the savings multiply (15% of original)? Why or why not? (Yes in the limit, but in practice the ToMe-merged tokens may *also* be the heavy hitters — so H2O picks them disproportionately. Net savings end up between 15% and 30%. This is the kind of empirical interaction your manager might want you to characterize.)

## My notes

(fill in as you go)

## References

- Chen et al., *An Image is Worth 1/2 Tokens After Layer 2 (FastV)*, ECCV 2024 Oral — arXiv:2403.06764.
- Bolya et al., *Token Merging: Your ViT But Faster (ToMe)*, ICLR 2023 — arXiv:2210.09461.
- Yang et al., *VisionZip*, CVPR 2025 — arXiv:2412.04467.
- Zhang et al., *H2O: Heavy-Hitter Oracle*, NeurIPS 2023 — arXiv:2306.14048.
- Xiao et al., *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)*, ICLR 2024 — arXiv:2309.17453.
