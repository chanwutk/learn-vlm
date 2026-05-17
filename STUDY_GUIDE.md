# VLM Pre-Internship Study Guide

A 16-day path from "I know transformers and basic CLIP" to "I can discuss the `SUGGESTED_READINGS.md` papers with my manager and brainstorm token-reduction methods."

Internship start: **2026-06-01**. Today: **2026-05-16**.

---

## How to use this guide

1. Open the topic file for **today's date** (see schedule below).
2. Work through the **Learning path** section — primary resource first, go-deeper only if you have time.
3. Do the **Self-check** at the bottom of the file. If you can't answer a question, the resource you used probably wasn't enough — try the go-deeper or fall back to the paper.
4. Anything that confuses you or sparks an idea → log it in [`MANAGER_QUESTIONS.md`](./MANAGER_QUESTIONS.md) with the date and module number.
5. Fill the **My notes** section in your own words. That's the bit that survives into the internship.

If a day slips, drop Module 08 first (it's the lightest) — the time goes to Module 07 catch-up.

---

## Table of contents

| # | Topic | File | Effort |
|---|-------|------|--------|
| 00 | Orientation & token-reduction taxonomy | [`notes/00-orientation.md`](./notes/00-orientation.md) | ~1 hr |
| 01 | Modern LLM internals (KV cache, RoPE, FlashAttn) | [`notes/01-llm-internals.md`](./notes/01-llm-internals.md) | ~3 hrs |
| 01b | Training pipelines (pre-train, SFT, RLHF, DPO, GRPO) | [`notes/01b-training-pipelines.md`](./notes/01b-training-pipelines.md) | ~3 hrs |
| 02 | Vision encoders for VLMs (ViT recap, CLIP, SigLIP) | [`notes/02-vision-encoders.md`](./notes/02-vision-encoders.md) | ~2 hrs |
| 03 | VLM architectures (LLaVA / BLIP-2 / Qwen-VL; projectors) | [`notes/03-vlm-architectures.md`](./notes/03-vlm-architectures.md) | ~5 hrs |
| 04 | High-resolution visual encoding (AnyRes, LLaVA-UHD, FastVLM) | [`notes/04-high-res-encoding.md`](./notes/04-high-res-encoding.md) | ~4 hrs |
| 05 | Video VLMs (frame sampling, token explosion) | [`notes/05-video-vlms.md`](./notes/05-video-vlms.md) | ~3 hrs |
| 06 | Token-reduction toolbox (FastV, ToMe, VisionZip, KV-cache) | [`notes/06-token-reduction-toolbox.md`](./notes/06-token-reduction-toolbox.md) | ~6 hrs |
| 07 | Reading-list deep dives (STORM, DyCoke, ESV, VideoBrain, VideoINSTA, Less-is-More) | [`notes/07-reading-deep-dives.md`](./notes/07-reading-deep-dives.md) | ~8 hrs |
| 08 | Evaluation (Video-MME, MVBench, EgoSchema, LongVideoBench, NExT-QA) | [`notes/08-evaluation.md`](./notes/08-evaluation.md) | ~2 hrs |

Modules **01, 03, 06** each contain a ~30-line PyTorch code exercise. The rest are theory + diagrams + self-check only.

---

## Day-by-day schedule (May 16 → May 31)

| Date | Modules | Why |
|------|---------|-----|
| **May 16 (Sat)** | 00 Orientation | Mental map of the field — placing every paper in a bucket before you read any of them. |
| **May 17 (Sun)** | 01 LLM internals | KV cache is the *reason* token reduction matters. Get this right and every other module is easier. |
| **May 18 (Mon)** | 01b Training pipelines | Pre-train → SFT → RLHF/DPO/GRPO. Every paper assumes you know this stack — fastest single ROI for the next two weeks. |
| **May 19 (Tue)** | 02 Vision encoders (ViT recap + CLIP + SigLIP) | You already know ViT/CLIP basics; this is one day to pick up SigLIP's sigmoid loss and the family lineage. |
| **May 20 (Wed)** | 03 VLM architectures (part 1: projectors + MLP/Q-Former/cross-attn) | The *connector* is where most token-reduction methods will live. |
| **May 21 (Thu)** | 03 VLM architectures (part 2: LLaVA family) | LLaVA is the modal-default of the open VLM world. |
| **May 22 (Fri)** | 03 VLM architectures (part 3: Qwen-VL family + BLIP-2/InternVL) | Lineage view: what changed and why between versions. |
| **May 23 (Sat)** | 04 High-res encoding (part 1: AnyRes + LLaVA-UHD) | Where the token count first explodes. |
| **May 24 (Sun)** | 04 High-res encoding (part 2: FastVLM) | Encoder-side efficiency story. |
| **May 25 (Mon)** | 05 Video VLMs (part 1: frame sampling) | Why naively stacking frames is hopeless. |
| **May 26 (Tue)** | 05 Video VLMs (part 2: temporal modeling + token-explosion math) | Sets up the next two modules. |
| **May 27 (Wed)** | 06 Token-reduction toolbox (part 1: pruning + merging — FastV, ToMe) | Two ideas you'll see in nearly every reading-list paper. |
| **May 28 (Thu)** | 06 Token-reduction toolbox (part 2: VisionZip, KV-cache compression, query-based) | Rounds out the taxonomy. |
| **May 29 (Fri)** | 07 Reading deep dives (part 1: STORM, DyCoke, Less-is-More) | Token-compression cluster. |
| **May 30 (Sat)** | 07 Reading deep dives (part 2: Efficient Video Sampling, VideoBrain, VideoINSTA) | Adaptive-sampling cluster. |
| **May 31 (Sun)** | 08 Evaluation + review | Benchmarks + skim everything once more; sharpen MANAGER_QUESTIONS.md for Day 1. |

---

## If you're falling behind — the minimum viable path

You'll probably skip something. Here's the priority order so you skip the *right* things.

**Spine (cannot cut, this is the conversation with your manager):**
- **07 Reading deep dives** — these *are* the reading list.
- **06 Token-reduction toolbox** — every reading-list paper builds on FastV and ToMe.
- **04 High-res encoding** — three reading-list papers live here (SigLIP, LLaVA-UHD, FastVLM).
- **03 VLM architectures** — needed to read LLaVA + Qwen-VL.
- **00 Orientation** — 1 hour; sets your mental map so 07 isn't terrifying.

**Cuttable in this order, if forced (shed the *bottom* first):**
1. **08 Evaluation** → 30-min skim of the benchmark table only.
2. **02 Vision encoders** → you know CLIP basics; skip the file, just read SigLIP's §3 directly.
3. **05 Video VLMs** → skim the token-count math + "two regimes" sections only.
4. **01b Training pipelines** → defer until you hit a paper using DPO/GRPO; the Karpathy chapters cover it in 75 min on demand.
5. **01 LLM internals** → if you already understand KV cache, the rest is reference.

**Paper priority within the 11:**
- 🔴 **Must read carefully** (manager will ask): all 6 Module-07 papers, LLaVA-UHD, FastVLM.
- 🟡 **Skim by skim path** (recognize, not reproduce): CLIP, LLaVA, Qwen-VL, SigLIP.

If today is May 30 and you've done nothing: read Module 00 (1 hr), then jump to Module 07 with skim paths. Better than no map at all.

---

## Progress tracker

Modules:
- [ ] 00 Orientation
- [ ] 01 LLM internals
- [ ] 01b Training pipelines
- [ ] 02 Vision encoders
- [ ] 03 VLM architectures
- [ ] 04 High-res encoding
- [ ] 05 Video VLMs
- [ ] 06 Token-reduction toolbox
- [ ] 07 Reading deep dives
- [ ] 08 Evaluation

Papers from `SUGGESTED_READINGS.md`:

*Foundational / Pioneering VLMs*
- [ ] CLIP (Radford et al., ICML 2021) — covered in Module 02
- [ ] LLaVA (Liu et al., NeurIPS 2023) — covered in Module 03
- [ ] Qwen-VL technical report (most fundamental release + deltas) — covered in Module 03
- [ ] SigLIP (Zhai et al., 2023) — covered in Modules 02 & 04

*High-resolution and efficient encoding*
- [ ] LLaVA-UHD (ECCV 2024) — covered in Module 04
- [ ] FastVLM (Apple, CVPR 2025) — covered in Module 04

*Token compression & reduction*
- [ ] STORM (NVIDIA, 2025) — covered in Module 07
- [ ] DyCoke (CVPR 2025) — covered in Module 07
- [ ] Less is More (COLING 2025) — covered in Module 07
- [ ] Efficient Video Sampling (arXiv 2024) — covered in Module 07

*Adaptive frame & spatial selection*
- [ ] VideoBrain (arXiv 2025) — covered in Module 07
- [ ] VideoINSTA (EMNLP Findings 2024) — covered in Module 07
