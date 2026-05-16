# 02 — Vision encoders for VLMs

## Why this matters

Every VLM you'll read about begins with the same step: turn pixels into a sequence of tokens that "live in the same space" as text. The encoder choice shapes (a) how many tokens are produced, (b) how good they are, and (c) whether they're aligned to language at all. CLIP and SigLIP between them account for the encoders in almost the entire reading list. Two days here gives you the prerequisites for Modules 03 (architectures) and 04 (high-res encoding).

## Prerequisites

Module 01 (KV cache picture). You already know basic transformer / ViT / CLIP, so much of this is targeted recap rather than first introduction.

## Learning path

### 1. ViT — the parts that matter for VLMs

You already know ViT splits an image into patches and runs a standard transformer. The two things to **lock in for VLM work**:
- The output of the encoder is a *sequence* of patch tokens (often plus a `[CLS]` token). VLMs typically use the patch tokens (not just `[CLS]`) so the LLM can attend to spatial regions. A 336×336 image at patch size 14 → **576 visual tokens**.
- Position encoding is at the encoder level only — the LLM downstream sees a sequence of "visual tokens" with no inherent 2D layout unless the architecture or the connector adds it back.

- **Primary:** [How the Vision Transformer (ViT) works in 10 minutes — AI Summer](https://theaisummer.com/vision-transformer/) — refresher with diagrams, ~10 min.
- **Go deeper:** [Yannic Kilcher — *An Image is Worth 16x16 Words*](https://www.youtube.com/watch?v=TrdevFK_am4) — paper walkthrough; only watch if the AI-Summer post left something unclear.

### 2. CLIP — contrastive pre-training, what survives into VLMs

You know the picture: two encoders (image, text), contrastive loss pulls matching pairs together and non-matching pairs apart, big web-scale data. **What matters for VLMs:**
- VLMs almost always reuse the CLIP **image encoder** (frozen or partially fine-tuned). The text tower is usually thrown away — the LLM replaces it.
- CLIP's image features are already roughly language-aligned, which is why a tiny MLP projector (LLaVA-style) is enough to make them readable by an LLM.
- "ViT-L/14 @ 336px" (576 tokens) is the default of the LLaVA family — note the token count; it's the baseline for Module 04's high-res story.

- **Primary:** [Samuel Albanie — CLIP digest](https://samuelalbanie.com/digests/2022-04-clip/) — concise paper digest plus video pointer.
- **Go deeper:** Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021 — [arXiv:2103.00020](https://arxiv.org/abs/2103.00020).

### 3. SigLIP — the open default encoder

What's new: replace the softmax-over-the-batch contrastive loss with a **per-pair sigmoid loss**. The softmax version requires comparing each image-text pair to *every other pair in the batch*; sigmoid treats each pair independently as a binary "match / no-match" classification. Consequences:
- Decouples performance from batch size. You can train with batches as small as a few thousand and still get high quality — softmax CLIP wanted ~32k batch size.
- Better calibration at the small-batch end → cheaper training and easier scaling experiments.
- The model itself is architecturally near-identical to CLIP; the win is in the loss.

In the SigLIP-family lineage:
- **SigLIP** (Zhai et al., 2023) — original sigmoid loss.
- **SigLIP 2** (Tschannen et al., 2025) — adds self-distillation and masked-prediction objectives on top; better fine-grained localization features. Several modern open VLMs (e.g., PaliGemma 2, Qwen2.5-VL) sit on SigLIP / SigLIP 2.

**Why this is in the *high-resolution* section of `SUGGESTED_READINGS.md`:** SigLIP scales cleanly to variable input resolution because the loss is not normalized per-batch — a key enabler for the AnyRes-style high-res encoders you'll meet in Module 04.

- **Primary:** [CLIP to SigLIP — Ritwik Raha blog](https://blog.ritwikraha.dev/choosing-between-siglip-and-clip-for-language-image-pretraining) — short side-by-side comparison.
- **Go deeper:** Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, 2023 — [arXiv:2303.15343](https://arxiv.org/abs/2303.15343). Especially §3 (the loss derivation) and §4.2 (batch-size scaling).

## Self-check

- **Why do VLMs use the patch tokens from a ViT instead of just the `[CLS]` token?** (LLMs benefit from spatial detail; a single CLS embedding throws away localization, which kills VQA-style tasks.)
- **State the softmax-CLIP loss and the SigLIP loss in one sentence each.** Softmax: each image's positive text is its match relative to all texts in the batch. Sigmoid: each (image, text) pair is independently labeled positive or negative.
- **Why does SigLIP train well at small batch sizes when CLIP doesn't?** (Sigmoid has no normalization across the batch, so the gradient signal per-pair doesn't depend on the batch as a denominator.)
- **A 336×336 image at patch 14 produces how many visual tokens? At patch 16?** (576 at p=14; 441 at p=16 = (336/16)² with appropriate rounding.) Internalize this number — it's the baseline in every reduction paper.
- **Mechanism question:** if you swapped a CLIP encoder for a SigLIP encoder inside LLaVA without retraining the projector, what would break — and why?

## My notes

(fill in as you go)

## References

- Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021 — arXiv:2010.11929.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML 2021 — arXiv:2103.00020.
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training (SigLIP)*, 2023 — arXiv:2303.15343.
- Tschannen et al., *SigLIP 2*, 2025 — arXiv:2502.14786.
