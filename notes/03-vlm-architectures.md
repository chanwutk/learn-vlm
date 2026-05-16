# 03 — VLM architectures

## Why this matters

This is the architectural core. Every token-reduction paper assumes a specific point in this pipeline to attack — connector vs. early LLM vs. KV cache. If you know the three canonical connector designs and the two-stage training recipe, you can read any open-VLM paper and immediately answer "what changed?" Three days here because there's a lot, but the *patterns* repeat — LLaVA's design is the template most others extend.

## Prerequisites

Modules 01 (KV cache) and 02 (CLIP / SigLIP encoders).

## Learning path

### 1. Connector designs — how visual tokens enter the LLM

Three canonical patterns. All modern VLMs are basically one of these three, sometimes a hybrid.

```
                                Visual tokens (long)
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       │                               │                               │
       ▼                               ▼                               ▼
  (A) MLP projector              (B) Q-Former                  (C) Cross-attention
                                                              + Perceiver Resampler
  Visual tok × MLP →           N learnable queries           Visual tokens act as K/V
  LLM-dim tokens.              cross-attend to visual         to extra cross-attn layers
  Concat with text             tokens → fixed N tokens.       inserted into a frozen LLM.
  → feed into LLM.             Feed those into LLM.
  Cost: O(N_img).              Cost: O(N_queries × N_img).    Cost: O(N_img) but in-place
  Examples: LLaVA,             Examples: BLIP-2,              in LLM layers.
  Qwen2.5-VL.                  InstructBLIP.                  Examples: Flamingo, IDEFICS.
```

Practical implications for token reduction:
- **MLP projector** = visual token count *stays* high; reduction methods (FastV, ToMe) operate inside the LLM. This is the dominant pattern today.
- **Q-Former** = visual token count is *capped* at the number of queries (e.g., 32). Reduction is "free" but you pay accuracy for tasks that need fine spatial detail.
- **Cross-attention** = visual tokens never get concatenated into the LLM sequence, so the KV-cache problem is sidestepped. Out of fashion now because pre-training is harder than concat-style.

### 2. Two-stage training (LLaVA recipe)

Almost every open VLM uses some variant of this:

| Stage | Trainable | Data | Goal |
|---|---|---|---|
| **1. Alignment** | only the projector (LLM + encoder frozen) | image-caption pairs (e.g., CC3M filtered) | Teach the projector to put visual tokens in the LLM's embedding space. |
| **2. Visual instruction tuning** | projector + LLM (encoder usually frozen) | GPT-4-generated image-conversation data + VQA + grounding | Teach the joint model to *follow visual instructions*. |

Why this matters for token reduction: the projector is the *learned* boundary between modalities. Token reduction methods that operate before the projector behave like input compression; methods that operate after operate like activation pruning. Training-free methods that work inside the LLM (FastV) implicitly bet that the projector has *already* produced redundant tokens.

- **Primary:** [LLaVA paper site](https://llava-vl.github.io/) — skim. The diagram + Table 1 (data composition) are the things to internalize.
- **Go deeper:** Liu et al., *Visual Instruction Tuning*, NeurIPS 2023 — [arXiv:2304.08485](https://arxiv.org/abs/2304.08485).

### 3. The LLaVA family lineage

| Version | What changed | Why |
|---|---|---|
| **LLaVA** (NeurIPS 2023) | First MLP-projector + visual-instruction-tuning recipe. ViT-L/14 @ 224. | Establishes the template. |
| **LLaVA-1.5** | ViT-L/14 @ **336**, larger projector (2-layer MLP), more instruction data. | Bigger encoder = more spatial detail. **Token count: 576.** |
| **LLaVA-NeXT (1.6)** | **AnyRes**: split a high-res image into multiple 336-tiles, encode each. Higher OCR / detail performance. **Token count: up to 2880.** | Token-explosion problem becomes acute → motivates Module 04. |
| **LLaVA-OneVision** | Same family extended to multi-image and video. Per-frame 196 tokens. | Same architectural ideas, video-ified. |
| **LLaVA-OneVision-1.5** | Open-data + cluster-discrimination encoder (RICE-ViT) instead of SigLIP/DFN. | Encoder swaps as the field evolves. |

- **Primary:** [LLaVA-OneVision blog post](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) — covers OV's place in the lineage; skim.
- **Go deeper:** Liu et al., *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)*, 2023 — [arXiv:2310.03744](https://arxiv.org/abs/2310.03744).

### 4. The Qwen-VL family lineage

This is the lineage your manager's reading list points at via "Qwen 3.5 technical report" (clarification pending — see `MANAGER_QUESTIONS.md`).

| Version | What changed | Why |
|---|---|---|
| **Qwen-VL** (Aug 2023) | Custom Vision Receptor + cross-attention adapter. 3-stage training: pretrain → multi-task pretrain → SFT. Adds grounding / OCR via box-coordinates in text. | Establishes the lineage; first widely-used non-LLaVA open VLM. |
| **Qwen2-VL** (Sep 2024) | **M-RoPE** (Multimodal RoPE) encoding (time, height, width) separately; native dynamic-resolution ViT. | Lets the model handle variable image sizes without resizing. |
| **Qwen2.5-VL** (Feb 2025) | Custom ViT with **2D-RoPE + window attention**; simple **MLP-based "vision-language merger"** (back to LLaVA-style). Pretrained at native variable resolution. | Reduces ViT compute cost; merger handles the long-feature problem. |
| **Qwen3-VL** (2025+) | Latest release; iterates on the merger and training data. | Field velocity. |

The pattern: across versions, the team migrated *toward* MLP-style merging (the LLaVA template won) and *added* dynamic-resolution + RoPE-axes-for-modality tricks.

- **Primary:** [Qwen2.5-VL technical blog](https://qwenlm.github.io/blog/qwen2.5-vl/) — short, has the architecture diagram.
- **Go deeper:** Bai et al., *Qwen-VL: A Versatile Vision-Language Model*, 2023 — [arXiv:2308.12966](https://arxiv.org/abs/2308.12966), then jump to Qwen Team, *Qwen2.5-VL Technical Report*, 2025 — [arXiv:2502.13923](https://arxiv.org/abs/2502.13923).

### 5. BLIP-2 and the Q-Former, in one read

You don't need the full BLIP-2 mechanics, just the Q-Former picture:
- A small transformer with **N learnable query tokens** (typically 32).
- Cross-attends to the (frozen) vision encoder output.
- Outputs exactly N visual tokens regardless of image size.
- This is **architectural** token reduction — built into the connector.

- **Primary:** [HuggingFace BLIP-2 blog](https://huggingface.co/blog/blip-2) — short.
- **Go deeper:** Li et al., *BLIP-2*, ICML 2023 — [arXiv:2301.12597](https://arxiv.org/abs/2301.12597).

InternVL is worth knowing exists: huge vision encoder (~6B) + MLP projector → LLM. Same template as LLaVA but with a heavyweight encoder.

## Code exercise — a minimal VLM forward pass (shapes only)

Goal: feel the dominance of visual tokens in the LLM input. ~25 lines.

```python
import torch
import torch.nn as nn

# Realistic CLIP-ViT-L/14 @ 336 numbers used by LLaVA-1.5.
N_IMG_TOK, D_VISION = 576, 1024           # 24*24 patches; CLIP-L hidden dim
D_LLM, T_TXT, B = 4096, 12, 1             # Vicuna-7B-style hidden; ~12-token prompt

class MLPProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out)
        )
    def forward(self, x):                  # x: (B, N, d_in)
        return self.net(x)                 # (B, N, d_out)

# Stand-ins for the real CLIP/SigLIP and the LLM's embedding table.
vision_features = torch.randn(B, N_IMG_TOK, D_VISION)
text_embeddings = torch.randn(B, T_TXT, D_LLM)

projector = MLPProjector(D_VISION, D_LLM)
visual_tokens = projector(vision_features)              # (B, 576, 4096)

# LLaVA-style "prefix": concatenate visual tokens then text tokens.
llm_inputs = torch.cat([visual_tokens, text_embeddings], dim=1)  # (B, 588, 4096)

share = N_IMG_TOK / (N_IMG_TOK + T_TXT) * 100
print("vision_features :", tuple(vision_features.shape))
print("visual_tokens   :", tuple(visual_tokens.shape))
print("text_embeddings :", tuple(text_embeddings.shape))
print("llm_inputs      :", tuple(llm_inputs.shape))
print(f"visual share    : {share:.1f}%  ← motivation for token reduction")
```

After running, sit with the last line: **~98% of the LLM input is visual.** Halving that does not halve quality; it roughly halves cost.

## Self-check

- **Sketch the three connector designs from memory** (MLP / Q-Former / cross-attention) and state the visual-token count each produces relative to N (the encoder output length).
- **Why is the projector frozen in stage 1 but trained in stage 2?** (Stage 1 aligns visual to text space; stage 2 fine-tunes the joint behavior. Doing both at once destabilizes alignment.)
- **What does AnyRes do to the token count of LLaVA-1.5?** (Multiplies it by ~5 in the typical 4-tile-plus-thumbnail layout.)
- **Why did Qwen2-VL introduce M-RoPE?** (To encode (time, height, width) as separate axes inside RoPE so the LLM can reason about position across modalities cleanly.)
- **Mechanism question:** if your projector is a 2-layer MLP and you apply ToMe-style merging **before** the projector vs. **after**, what's different? (Before: you save projector compute too, but the merge sees only raw vision-encoder features. After: you save downstream LLM compute but pay full projector compute. Most papers do "after" for a reason — vision features alone are harder to compare.)

## My notes

(fill in as you go)

## References

- Liu et al., *Visual Instruction Tuning (LLaVA)*, NeurIPS 2023 — arXiv:2304.08485.
- Liu et al., *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)*, 2023 — arXiv:2310.03744.
- Li et al., *LLaVA-OneVision*, 2024 — arXiv:2408.03326.
- Bai et al., *Qwen-VL*, 2023 — arXiv:2308.12966.
- Wang et al., *Qwen2-VL*, 2024 — arXiv:2409.12191.
- Qwen Team, *Qwen2.5-VL Technical Report*, 2025 — arXiv:2502.13923.
- Li et al., *BLIP-2*, ICML 2023 — arXiv:2301.12597.
- Alayrac et al., *Flamingo*, NeurIPS 2022 — arXiv:2204.14198.
