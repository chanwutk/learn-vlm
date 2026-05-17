# 07 — Reading-list deep dives

## Why this matters

This is the day-1-conversation-with-your-manager module. Six papers, two days. By now (after Modules 00–06) you should know the field, the architectures, and the toolbox. Here you fill in the specifics of each reading-list paper. Each section is intentionally a **skeleton** — your job is to fill in the answers as you read. Don't restate abstracts; aim for three crisp sentences per cell.

## Prerequisites

All previous modules.

## How to fill each section

For each paper, the spine is **problem → key idea → method → results → why it matters for token reduction**, per `CLAUDE.md`. Plus one row for **what to ask your manager** — put the most interesting open question per paper there so it can graduate into `MANAGER_QUESTIONS.md`.

---

## 1. STORM — Token-Efficient Long Video Understanding

**Citation:** Jiang et al., NVIDIA, *STORM: Token-Efficient Long Video Understanding for Multimodal LLMs* — [arXiv:2503.04130](https://arxiv.org/abs/2503.04130). Project page: [research.nvidia.com/labs/lpr/storm](https://research.nvidia.com/labs/lpr/storm/).

**Bucket (from Module 00):** Merging + temporal compression, **trained**, lives between encoder and LLM as a dedicated temporal projector.

**Skim path (if you only have 30 minutes):** §3 (Method — the three components: Mamba temporal projector, temporal compression, spatial compression) + Figure 1 (architecture). Then §5 results table — focus on the budget-vs-accuracy *curve*, not the absolute SOTA numbers.

#### Quick detour: what's Mamba / SSM?

You'll hit "Mamba-based temporal projector" in STORM. Mamba is a **State Space Model (SSM)** — a sequence model that's an alternative to attention. The picture:

```
   Attention (Transformer)            SSM (Mamba)
   --------------------------         --------------------------
   For each token, look at all        Maintain a hidden state h_t
   previous tokens via Q·K^T.         that updates recurrently:
                                          h_t = Ā h_{t-1} + B̄ x_t
   Cost: O(T²) compute,                   y_t = C h_t
         O(T) KV cache.              Cost: O(T) compute, O(1) state.

                                     "Selective" (Mamba):
                                     Ā, B̄ become input-dependent so
                                     the model can *choose* what to
                                     keep in state vs. forget.
```

| | Attention | Mamba (SSM) |
|---|---|---|
| Sequence-length scaling | Quadratic compute, linear memory | Linear compute, constant memory |
| Long-range expressivity | Strong — can attend anywhere | Strong if the selection mechanism is good (the Mamba contribution) |
| Trained? | Yes | Yes — different architecture, same loss types |

**Why STORM uses it:** the temporal sequence of frame tokens is *long*. Attention over hundreds of frames is brutally expensive. A Mamba-based temporal projector integrates information across many frames *linearly*, then hands a compressed summary to the (still-attention-based) LLM.

You don't need to read the Mamba paper, but if curious: Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023 — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). **5-min skim:** §1 + Figure 1 only.

| Cell | Notes |
|---|---|
| **Problem** | Image-VLM-style architectures treat frames independently → no explicit temporal modeling, and the per-frame token explosion makes long-video impractical. |
| **Key idea** | (fill in) Hint: insert a **Mamba-based temporal encoder** between vision encoder and LLM, then compress in *both* time and space. |
| **Method** | (fill in) Three components — temporal projector, temporal compression (training-free + training-based), spatial compression. Draw the block diagram in *My notes*. |
| **Results** | (fill in) > 5% on MLVU & LongVideoBench, with reduced compute. Note which baselines they beat and at what token budget. |
| **Why it matters for token reduction** | First major architecture that puts a **state-space model** (Mamba) in the connector for video VLMs — sidesteps quadratic attention over the time axis. |
| **Ask my manager** | (fill in) e.g., *"Is Mamba in the connector actually load-bearing here, or would a small transformer do the same?"* |

---

## 2. DyCoke — Dynamic Compression of Tokens for Fast Video Large Language Models

**Citation:** Tao et al., *DyCoke*, CVPR 2025 — [arXiv:2411.15024](https://arxiv.org/abs/2411.15024). Code: [github.com/KD-TAO/DyCoke](https://github.com/KD-TAO/DyCoke).

**Bucket:** Merging + KV-cache compression, **training-free**, in prefill *and* decode.

**Skim path (if you only have 30 minutes):** §3 (Method — the two-stage pipeline: temporal token merging in prefill + dynamic KV-cache pruning in decode) + Figure 2 (which tokens get dropped where). The algorithm boxes are short — read them line by line. Skip §4 ablations on first pass.

| Cell | Notes |
|---|---|
| **Problem** | One-shot pruning fails for video because different frames are attended to at different decode steps; static spatial pruning drops tokens that mattered for *later* generated text. |
| **Key idea** | (fill in) Hint: **two-stage**: (a) temporal token merging in prefill, (b) dynamic KV-cache pruning in decode. |
| **Method** | (fill in) Stage 1: TTM merges similar tokens across nearby frames (ToMe-style). Stage 2: KV-cache pruning drops the least-attended cached tokens *per-step*, recovering some if the new step pays attention to them. |
| **Results** | (fill in) Reported 1.5× speedup and 1.4× memory reduction at improved task performance, no training. |
| **Why it matters for token reduction** | Cleanest example of "merge in prefill + evict in decode" — covers both halves of the inference cost. The dynamic per-step decision in stage 2 is the unusual bit; most KV-cache methods commit once. |
| **Ask my manager** | (fill in) e.g., *"How does DyCoke's dynamic KV pruning compose with H2O's heavy-hitter rule? Are they fighting or stacking?"* |

---

## 3. Less is More — TRIM: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs

**Citation:** Song et al., *Less is More (TRIM)*, COLING 2025 — [arXiv:2409.10994](https://arxiv.org/abs/2409.10994).

**Bucket:** Pruning / selection, **training-free**, at the connector / early LLM boundary; CLIP-similarity-driven.

**Skim path (if you only have 20 minutes):** §3 (Method — the CLIP-similarity scoring rule, ~2 pages) + the side-by-side table comparing TRIM vs. FastV and baselines. The whole paper is small, but the §3 procedure is the only thing you actually need to be able to discuss.

| Cell | Notes |
|---|---|
| **Problem** | Most token-reduction methods either need fine-tuning or rely on a learned scorer; can we get FastV-quality results with even simpler ingredients? |
| **Key idea** | (fill in) Hint: use the **CLIP similarity between each visual token and the text** as the importance metric. No new parameters, no fine-tune. |
| **Method** | (fill in) Compute per-token text–image similarity; keep top-K; pass only those to the LLM. They call this TRIM (Token Reduction using CLIP Metric). |
| **Results** | (fill in) Strong performance across 12 datasets with substantial compute drop. Compare to FastV's 45% FLOPs figure. |
| **Why it matters for token reduction** | A clean *text-conditioned* training-free baseline. Useful as the comparison point when you propose anything text-aware. |
| **Ask my manager** | (fill in) e.g., *"Does TRIM beat FastV because text-conditioning > LLM-attention-conditioning, or because CLIP similarity is just a better proxy than late-layer LLM attention?"* |

---

## 4. Efficient Video Sampling (EVS) — Pruning Temporally Redundant Tokens

**Citation:** *Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference* — [arXiv:2510.14624](https://arxiv.org/abs/2510.14624).

**Bucket:** Temporal pruning at the patch level, **training-free**, encoder-adjacent.

**Skim path (if you only have 20 minutes):** §3 (Method — the static-patch detection rule with threshold *q*) + Figure 1 (what a "temporally static patch" looks like) + the §4 table showing TTFT speedup at different *q*. The intro alone gives you 70% of the paper.

| Cell | Notes |
|---|---|
| **Problem** | In long video (CCTV / lectures / driving footage), most spatial patches are nearly identical across consecutive frames — they encode no new information yet are tokenized and attended to fully. |
| **Key idea** | (fill in) Hint: prune **temporally static patches** — patches whose pixel/feature values change less than a threshold across nearby frames. |
| **Method** | (fill in) Per-patch motion / change detector; drop static patches; keep positional identity intact so the LLM still knows where the kept ones are. |
| **Results** | (fill in) Up to 4× faster time-to-first-token at q=0.75 (75% static-redundancy threshold), minimal accuracy hit. |
| **Why it matters for token reduction** | A *patch-level temporal* reducer — orthogonal to most spatial methods. Trivial to compose with FastV / ToMe. |
| **Ask my manager** | (fill in) e.g., *"For action-heavy video (where most patches change), does EVS degrade gracefully or fall off a cliff?"* |

---

## 5. VideoBrain — Learning Adaptive Frame Sampling for Long Video Understanding

**Citation:** *VideoBrain* — [arXiv:2602.04094](https://arxiv.org/abs/2602.04094).

**Bucket:** Adaptive frame sampling, **trained** (policy learned via RL-style reward), pre-encoder.

**Skim path (if you only have 30 minutes):** §3 (Method — dual-agent design: CLIP retrieval agent + uniform agent) + §4 (the **behavior-aware reward function** — this is the novelty) + the headline table comparing frame budgets vs. accuracy. Skip the agent-tool implementation details; you mostly need the reward-function picture. Cross-link: this is where Module 01b's RL/GRPO content pays off.

| Cell | Notes |
|---|---|
| **Problem** | Uniform / single-shot keyframe selection can't recover from poor early choices on long videos. We need a policy that can *call back* for more frames if it isn't sure yet. |
| **Key idea** | (fill in) Hint: an end-to-end *agentic VLM* that decides whether to invoke a frame-fetching agent (CLIP-based retrieval or uniform sampling) before answering. Reward function penalizes over-calling. |
| **Method** | (fill in) Two complementary agents (CLIP retrieval, uniform); the VLM directly perceives frames and reasons about *information sufficiency*. A behavior-aware reward discourages reward-hacking ("just always call agents"). |
| **Results** | (fill in) Reported gains over Qwen3-VL-8B-Instruct: +8.1% LongVideoBench, +9.0% LVBench, with **30–40% fewer frames on average**. |
| **Why it matters for token reduction** | Quite different from the rest of the list — this is a *policy* method, not a static reducer. The frame budget itself becomes the optimization variable; everything else is downstream. |
| **Ask my manager** | (fill in) e.g., *"How brittle is the behavior-aware reward — does it transfer across base VLMs, or is it tuned to Qwen3-VL?"* |

---

## 6. VideoINSTA — Zero-shot Long Video Understanding via Informative Spatial-Temporal Reasoning with LLMs

**Citation:** Liao et al., *VideoINSTA*, EMNLP Findings 2024 — [arXiv:2409.20365](https://arxiv.org/abs/2409.20365). Code: [github.com/mayhugotong/VideoINSTA](https://github.com/mayhugotong/VideoINSTA).

**Bucket:** Text-conditioned spatio-temporal aggregation, **trained / agentic** (no fine-tune of the VLM itself, but builds an inference-time pipeline), zero-shot framework.

**Skim path (if you only have 30 minutes):** §3 (Method — the three components: zero-shot framework, event-temporal + content-spatial reasoning, self-reflective sufficiency) + Figure 2 (the pipeline). The C-DPCKNN event-segmentation step is the technical core — everything else is orchestration logic. Compare its "sufficiency check" to VideoBrain's reward.

| Cell | Notes |
|---|---|
| **Problem** | Long-video reasoning needs the *right* information, not all of it; how to let an LLM aggregate only the informative spatial-temporal cues, zero-shot? |
| **Key idea** | (fill in) Hint: event-based temporal reasoning (auto-segment via C-DPCKNN clustering) + content-based spatial reasoning + a *self-reflective sufficiency check*. |
| **Method** | (fill in) Three components: (1) zero-shot LLM framework, (2) event-temporal + content-spatial reasoning, (3) self-reflective info-sufficiency scheme balancing temporal factors. |
| **Results** | (fill in) New SOTA on EgoSchema, NextQA, IntentQA, ActivityNetQA, zero-shot. |
| **Why it matters for token reduction** | The "self-reflective sufficiency" idea overlaps with VideoBrain's information-sufficiency reward — same intuition, very different implementation (training-free framework vs. trained policy). Comparing how these two pose the same question is interesting. |
| **Ask my manager** | (fill in) e.g., *"VideoINSTA and VideoBrain both ask 'is my information enough?' — how do they differ in failure modes, and which is closer to what we'd build first?"* |

---

## Cross-paper synthesis (write after finishing all six)

Once all six skeletons are filled, answer these in your own words (these become brainstorm fodder for `MANAGER_QUESTIONS.md`):

1. **Stack the buckets.** Each paper hits one or two of {temporal pruning, spatial pruning, merging, KV-cache, frame selection}. Make a table: paper × bucket. Which combinations are crowded? Which are empty?
2. **Training-free vs. trained — who wins?** STORM and VideoBrain are trained; the rest are training-free. Are the trained ones uniformly better, or is there a regime (e.g., extreme reduction) where they pull ahead?
3. **The "what's the right information?" question.** VideoBrain (trained agent) and VideoINSTA (zero-shot framework) converge on the same intuition. Are there other ways to ask this question — e.g., a tiny gating network that's neither agentic nor framework-based?
4. **Pick the paper you'd reproduce first** if your manager said "we need a baseline by next week." Why?

## My notes

(fill in per-paper as you read)

## References

- Jiang et al., *STORM*, 2025 — arXiv:2503.04130.
- Tao et al., *DyCoke*, CVPR 2025 — arXiv:2411.15024.
- Song et al., *Less is More (TRIM)*, COLING 2025 — arXiv:2409.10994.
- *Efficient Video Sampling (EVS)*, 2025 — arXiv:2510.14624.
- *VideoBrain*, 2026 — arXiv:2602.04094.
- Liao et al., *VideoINSTA*, EMNLP Findings 2024 — arXiv:2409.20365.
