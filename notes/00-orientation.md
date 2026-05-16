# 00 — Orientation & token-reduction taxonomy

## Why this matters

Before reading any paper, you want a mental map of where it lives. Otherwise every paper feels like a new universe. This module gives you (a) the field's vocabulary, (b) a taxonomy of token-reduction methods, and (c) a bucket label for every paper in `SUGGESTED_READINGS.md`. After this, you can talk about the field without having read any of it — which makes the actual reading much easier.

## Prerequisites

None. Start here.

## The one-paragraph picture

A **vision-language model (VLM)** takes an image (or video) and a text prompt and produces text. Internally: a **vision encoder** (usually a ViT, often CLIP- or SigLIP-pretrained) turns the image into a sequence of **visual tokens**. A **connector** (a.k.a. projector) maps those tokens into the LLM's embedding space. The LLM then runs decoder-style generation over `[visual tokens] + [text tokens]`. The number of visual tokens grows fast: a single 336×336 image at patch size 14 is 576 tokens; a 4-tile high-res image is 2304; **a 1-minute video at 8 fps is roughly 276,000**. That blow-up is what makes attention quadratic-cost prohibitive — and is the entire reason your internship topic exists.

```
        image / video
              │
              ▼
      ┌──────────────┐
      │ vision       │   (ViT, CLIP, SigLIP…)
      │ encoder      │
      └──────┬───────┘
             │  visual tokens (lots of them!)
             ▼
      ┌──────────────┐
      │ connector /  │   (MLP, Q-Former, cross-attn…)
      │ projector    │   ←─── where most token reduction lives
      └──────┬───────┘
             │  visual tokens in LLM embedding space
             ▼
      ┌──────────────┐
      │     LLM      │   ←─── KV cache memory grows here, also where
      │  (decoder)   │        prune/merge during inference can save FLOPs
      └──────┬───────┘
             ▼
            text
```

## Taxonomy of visual token reduction

Five buckets. Almost every paper you'll read is a combination of these.

| Bucket | Idea | Representative methods |
|---|---|---|
| **Pruning** | Drop tokens deemed unimportant (by attention score, similarity to text, or a learned scorer). | FastV, FasterVLM, VisionZip (selection part) |
| **Merging** | Combine similar tokens into one (often bipartite matching). | ToMe, DyCoke, VisionZip (merge part) |
| **Pooling / down-sampling** | Average / strided pool a fixed grid. Crude but free at inference. | Q-Former pre-pooling, average pooling baselines |
| **Query-based selection** | Learn (or condition on text) a small set of queries that gather information from many tokens. | Q-Former, Perceiver-IO-style, VideoINSTA-ish text-conditioned aggregation |
| **KV-cache compression** | At inference, drop / quantize / merge K-V entries already in the cache. | H2O, StreamingLLM, DyCoke (KV-side step), token-eviction methods |

Two orthogonal axes:
- **Training-free vs. trained.** Training-free methods drop into a pretrained VLM (FastV, ToMe). Trained methods learn a scorer/router (VideoBrain).
- **Where in the pipeline.** Encoder-side (before tokens enter the LLM — cheaper but blind to the question), prefill-side (early LLM layers), decode-side (during generation — KV-cache compression).

When reading a paper, ask: **which bucket(s), training-free or trained, and where in the pipeline?** That alone gets you 60% of the way to discussing it.

## Where each `SUGGESTED_READINGS.md` paper lands

| Paper | Bucket | Where | Trained? |
|---|---|---|---|
| CLIP | (foundational, not reduction) | encoder | — |
| LLaVA | (foundational, not reduction) | architecture | — |
| Qwen-VL | (foundational, not reduction) | architecture | — |
| SigLIP | (foundational encoder + appears in efficient-encoding work) | encoder | — |
| LLaVA-UHD | spatial decomposition + token compress | encoder→connector | trained |
| FastVLM | efficient encoder design (FastViTHD) | encoder | trained |
| STORM | merging + temporal compression | connector / between encoder & LLM | trained |
| DyCoke | merging + KV-cache compression | prefill + decode | training-free |
| Less is More | pruning / selection | early LLM layers | training-free |
| Efficient Video Sampling | temporal pruning (frame & token level) | encoder-side / frame-level | training-free |
| VideoBrain | adaptive frame sampling (query-aware) | pre-encoder | trained |
| VideoINSTA | text-conditioned spatio-temporal aggregation | between encoder & LLM | trained / agentic |

Reading-list insight: most of the *spatial* methods sit at the connector or in early LLM layers, while *temporal* methods sit at the frame-selection stage. Combining both is an underexplored direction — exactly the kind of seed for your goal #2 brainstorm.

## Learning path

1. **The "VLM 101" mental model** — read this file. The diagram + paragraph above is enough.
2. **A community-maintained taxonomy** — skim the README of the awesome-list below. You're looking for category names and how papers cluster; not reading any single paper yet.
   - **Primary:** [Awesome-Token-Compress (GitHub list, ViT + VLM token compression)](https://github.com/daixiangzi/Awesome-Token-Compress) — just skim section headings, 10 min.
   - **Go deeper:** [Awesome-Collection-Token-Reduction (broader, includes NLP)](https://github.com/ZLKong/Awesome-Collection-Token-Reduction)

## Self-check

Before moving on, you should be able to answer these without looking back:

1. **Where in a VLM pipeline do visual tokens first get created? Where do they cause the most cost?** (Hint: created at the encoder; cost lives in the LLM attention — quadratic in total tokens.)
2. **Name the five buckets of token reduction in your own words.** Pruning, merging, pooling, query-based selection, KV-cache compression.
3. **Pick any paper from the table above. State its bucket, where it sits, and whether it's trained.** Do this for at least three papers.
4. **What's the difference between encoder-side and decode-side reduction?** Roughly: encoder-side reduces tokens *before* the LLM sees them (cheap, fast, but loses information the question would have needed); decode-side reduces tokens already in the KV cache (expensive setup, smart targeting).
5. **Open question for the brainstorm log:** which two buckets above are most often *combined* in a single paper, and which combination is rare?

## My notes

(fill in as you go)

## References

- See `SUGGESTED_READINGS.md` for the full reading list. arXiv IDs collected in each module's References section.
