# 04 — High-resolution visual encoding

## Why this matters

CLIP-ViT-L/14 was trained at 224 or 336 pixels. Real-world VQA / OCR / chart / document tasks want **way** more resolution. The simplest fix (resize the encoder upward) doesn't work — pretrained position encodings don't extrapolate. The community's answer is "tile and re-encode": split a high-res image into multiple encoder-friendly tiles, encode each, *and now you have 5× the visual tokens you started with*. This module is about (a) the tiling tricks and (b) the encoder-side compression methods that fight back. Three of your suggested-readings papers live here: SigLIP, LLaVA-UHD, FastVLM.

## Prerequisites

Modules 02 (CLIP/SigLIP) and 03 (LLaVA-NeXT mention of AnyRes).

## The pixel-count to token-count problem

```
                  CLIP-ViT-L/14 @ 336
   Native            (n_img_tok = 576)
   resolution               ─────► OK for a thumbnail VQA.
   336×336

                  AnyRes (LLaVA-NeXT) — split into 2×2 + thumbnail
   High-res
   672×672      ┌─┬─┐       4 tiles × 576 + 1 thumb × 576
                │█│█│  ──►  = 5 × 576 = 2880 tokens
                ├─┼─┤
                │█│█│
                └─┴─┘
                                        ─────► Token count blew up 5×.
```

That blow-up is the entire reason this module exists.

## Learning path

### 1. AnyRes — the dominant tiling trick

Algorithm:
1. Pick the closest "grid" (2×2, 1×3, 3×1, …) to the image's aspect ratio from a small fixed set.
2. Resize the image so each grid cell is the encoder's native size (e.g., 336×336).
3. Also resize the whole image to one encoder-native thumbnail.
4. Encode each tile + thumbnail independently with the same vision encoder.
5. Concatenate all tile tokens + thumbnail tokens → feed to projector.

Why it works: tiles preserve high-frequency detail (OCR), thumbnail preserves global layout. Why it's expensive: 4–6× the token count. Implication for reduction: every subsequent paper that brags about "X% fewer tokens" is implicitly fighting AnyRes' bloat.

- **Primary:** [LLaVA-NeXT blog (the original AnyRes write-up)](https://llava-vl.github.io/blog/2024-01-30-llava-next/) — read the AnyRes section, ~10 min.
- **Go deeper:** [LLaVA-NeXT ablations blog](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/) — what happens when you scale grids 2×2 → 6×6.

### 2. LLaVA-UHD — image slicing + compression done right

What it adds beyond AnyRes:
- **Adaptive variable-sized slices** instead of a fixed grid set. Selects the partition that best matches the image's aspect ratio without distortion.
- **A compression module** that condenses the per-slice tokens *before* the LLM sees them.
- A **spatial schema** (positional markers telling the LLM where each slice came from in the original layout).

Result reported in the paper: supports 672×1088 images (6× more pixels than LLaVA-1.5) **using 94% of the inference compute** of LLaVA-1.5 — i.e., higher res for slightly less cost — and +6.4 accuracy on TextVQA.

The takeaway for your internship: **encoder-side compression + smarter slicing is a real lever; this paper is one of the cleanest demonstrations.**

- **Primary:** [LLaVA-UHD v3 README (slicing + compression overview)](https://github.com/thunlp/LLaVA-UHD) — short, with figures.
- **Go deeper:** Xu et al., *LLaVA-UHD*, ECCV 2024 — [arXiv:2403.11703](https://arxiv.org/abs/2403.11703). Pay attention to §3 (image modularization) and §4 (compression module).

### 3. FastVLM — change the encoder, not the connector

Most papers in this module compress tokens *after* the encoder. FastVLM (Apple, CVPR 2025) instead **redesigns the encoder** to produce fewer tokens at higher resolution from the start.

Key ideas:
- **FastViTHD** — a hybrid conv + transformer encoder. Convolutions downsample early (cheap, no quality hit on natural images); transformer layers operate on the smaller token grid.
- ~8× smaller and ~20× faster than ViT-L/14 at comparable quality on VLM tasks.
- **Token reduction is "free"**: it's a property of the encoder architecture, not a post-hoc pruner. The team explicitly argues "scaling resolution + an efficient encoder beats token pruning" — a position you'll want to be able to discuss.
- The smallest variant beats LLaVA-OneVision-0.5B with 85× faster time-to-first-token.

For brainstorming (goal #2): FastVLM stakes out one extreme — push compression into the encoder. The reading-list papers in Module 07 mostly stake out the other — keep the encoder, compress later. The synthesis ("efficient encoder *and* late-stage reducer") is a candidate research direction.

- **Primary:** [Apple ML Research — FastVLM (project page)](https://machinelearning.apple.com/research/fast-vision-language-models) — short with figures.
- **Go deeper:** Vasu et al., *FastVLM*, CVPR 2025 — [arXiv:2412.13303](https://arxiv.org/abs/2412.13303). Read §3 (architecture) and §5 (ablations).

## Putting it together

A useful mental table — three knobs you can turn at the encoder boundary:

| Knob | Method | Trade-off |
|---|---|---|
| **Input image** | Higher resolution + tiling | More tokens, more detail. AnyRes / LLaVA-UHD. |
| **Encoder architecture** | Hybrid / efficient encoder that natively emits fewer tokens | Less flexibility, but free at inference. FastVLM. |
| **Connector-side compression** | Compress tokens between encoder and LLM | Plug-in; works on top of any encoder. LLaVA-UHD's compression module; many reading-list papers. |

Each reading-list paper is one or two of these knobs turned.

## Self-check

- **What problem does AnyRes solve, and what problem does it create?** (Solves: low-resolution input. Creates: 4–6× more visual tokens.)
- **How is LLaVA-UHD's slicing different from AnyRes?** (Variable-sized adaptive slicing instead of a fixed 2×2 / 1×3 set; preserves aspect ratio without distortion.)
- **Why does FastVLM argue we *don't* need token pruning?** (Because it solves the same problem upstream — an efficient encoder emits fewer tokens by design, so there's nothing left to prune.)
- **Mechanism question:** if you put FastVLM's FastViTHD encoder behind LLaVA-UHD's compression module, would you expect the gains to *stack*, *cancel*, or be roughly *redundant*? Argue both sides. (Stack: they operate on different bottlenecks — encoder compute vs. LLM tokens. Redundant: both reduce tokens, so the second one has fewer tokens to remove. The realistic answer: gains stack but with diminishing returns; this is exactly the kind of empirical question your manager might want you to think about.)
- **One number to remember:** how many tokens does an AnyRes 2×2 + thumbnail produce on a 336-token-per-tile encoder? (≈ 2880.)

## My notes

(fill in as you go)

## References

- Zhai et al., *SigLIP*, 2023 — arXiv:2303.15343 (also in Module 02).
- Xu et al., *LLaVA-UHD*, ECCV 2024 — arXiv:2403.11703.
- Xu et al., *LLaVA-UHD v2*, 2024 — arXiv:2412.13871.
- Vasu et al., *FastVLM*, CVPR 2025 — arXiv:2412.13303.
- LLaVA-NeXT team blog, *Improved reasoning, OCR, and world knowledge*, Jan 2024 (no arXiv; see references in Module 03).
