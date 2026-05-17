# 05 — Video VLMs

## Why this matters

Video makes Module 04's token-explosion problem dramatically worse — every second adds another image's worth of tokens. This module gives you (a) the math of why naive frame-stacking is hopeless, (b) the standard video-VLM patterns, and (c) frame-sampling tricks every Module 07 paper assumes are already in place.

## Prerequisites

Modules 03 (VLM architectures), 04 (AnyRes / high-res token explosion).

## The token-count math

```
   1 image, 336×336, ViT-L/14     =     576 tokens
   AnyRes 2×2 + thumb            ≈   2 880 tokens
   1 second of video @ 8 fps      ≈   4 608 tokens
   1 minute     @ 8 fps           ≈ 276 480 tokens
   1 hour       @ 1 fps (frugal!) ≈ 2 073 600 tokens     ← LLMs choke long before this
```

Even ignoring quality, this is a **memory** wall (KV cache linearly grows with sequence length) and a **compute** wall (prefill attention is quadratic). Every video VLM, without exception, sits on top of some combination of (i) low frame rate, (ii) per-frame token compression, and (iii) frame selection. The reading-list papers (STORM, DyCoke, VideoBrain, VideoINSTA, Efficient Video Sampling) attack one or more of these.

## Learning path

### 1. Frame sampling — the cheapest knob

Standard approaches, from crudest to smartest:
- **Uniform sampling** (k frames per video). Default baseline. Misses fast events; wastes frames on static segments. Used by Video-ChatGPT, Video-LLaVA, etc.
- **FPS-based sampling** (fixed rate, e.g., 1 fps). Same problem.
- **Scene-aware / motion-based sampling.** Sample more in segments with high optical flow or scene change. Cheap, training-free, sometimes good enough.
- **Question-conditioned (adaptive) sampling.** Use the text prompt to decide which frames to keep — relevant to your question, drop the rest. This is what VideoBrain learns; VideoINSTA does an agentic version.

**Why it matters:** every other technique downstream assumes a *good enough* set of frames. Bad sampling = no amount of clever token compression can recover the dropped information.

- **Primary:** Tang et al., *Video Understanding with Large Language Models: A Survey* — [arXiv:2312.17432](https://arxiv.org/abs/2312.17432). Skim the **frame-sampling** and **architecture** sections only — gives the lay of the land in 20–30 min.

### 2. Per-frame encoding and temporal modeling

Three patterns dominate:

| Pattern | How time is encoded | Where token reduction lives | Examples |
|---|---|---|---|
| **Per-frame ViT + temporal pooling** | Encode each frame separately; pool across time (avg / Q-Former). | Pooling is *the* reduction. | Video-ChatGPT (avg pool across time), Video-LLaMA (video Q-Former). |
| **Per-frame ViT + concat** | Encode each frame separately; concatenate all frame tokens. | Has to be applied post-hoc — KV blows up. | Naive Video-LLaVA-style, LLaVA-OneVision-Video. |
| **Joint spatio-temporal encoder** | One encoder sees a clip of frames as one input (3D-RoPE etc.). | Inside the encoder. | Qwen2-VL / Qwen2.5-VL video mode (M-RoPE). |

Pattern 2 is the most expressive (and most expensive). It's also the *most common target* of the reading list's token-reduction papers — they assume you've chosen pattern 2 and then attack the resulting token count.

Reuse from Module 03: most video VLMs are just an image VLM with a frame loop bolted on. The hard part isn't the architecture; it's deciding which frames + which tokens to keep.

- **Primary:** [HuggingFace Video-LLaVA docs](https://huggingface.co/docs/transformers/main/model_doc/video_llava) — quick read of the architecture.
- **Go deeper:** Lin et al., *Video-LLaVA*, 2023 — [arXiv:2311.10122](https://arxiv.org/abs/2311.10122). Quick: §3 (architecture) is what matters.

### 3. Token-explosion math, in numbers

```
Setup                                per-frame   × frames   = total
  Video-ChatGPT (avg-pool)             100        × 100      =   10 000
  Video-LLaVA (8 frames, full)         576        ×   8      =    4 608
  LLaVA-OneVision-Video (196/frame)    196        ×  32      =    6 272
  AnyRes-ish per frame                 2 880      ×  32      =   92 160   ← infeasible
  Long-video benchmark target          ~32 frames over a 10-min video — clearly need help.
```

### 4. The two regimes the reading list attacks

| Regime | Goal | Reading-list papers |
|---|---|---|
| **Mid video** | Fit a 30-frame video into one forward pass without quality loss. | DyCoke, "Less is More", STORM |
| **Long video** | Find the right ~30 frames in the first place. | VideoBrain, Efficient Video Sampling, VideoINSTA |

## Self-check

- **Name the three temporal-modeling patterns and a paper that uses each.** (Per-frame + pool: Video-ChatGPT. Per-frame + concat: Video-LLaVA. Joint spatio-temporal: Qwen2-VL.)
- **Why is uniform frame sampling a bad default for question-answering on a 10-minute video?** (The answer probably lives in a few seconds out of 600; uniform spends most of its budget on irrelevant frames.)
- **Sketch the token-count math for a 1-minute clip at 8 fps using a 576-token-per-frame encoder.** (8 × 60 × 576 ≈ 276 k tokens.)
- **What does the reading list's `Efficient Video Sampling` attack — frames or tokens?** (Both — it prunes *temporally* redundant tokens; covered in Module 07.)
- **Mechanism question:** if you had a perfect adaptive frame sampler (always returns the right N frames), would you still need per-frame token reduction? Why? (Yes, because each frame's *spatial* tokens are still redundant — most of the frame is background. The two methods compose.)

## My notes

(fill in as you go)

## References

- Lin et al., *Video-LLaVA*, 2023 — arXiv:2311.10122.
- Tang et al., *Video Understanding with Large Language Models: A Survey* — arXiv:2312.17432.
