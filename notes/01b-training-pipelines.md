# 01b — Training pipelines for LLMs and VLMs

## Why this matters

Every paper in `SUGGESTED_READINGS.md` assumes you already know the pre-train → SFT → RL post-training stack. Without it, sentences like "instruction-tuned on multi-turn dialogues", "aligned with DPO on a 10k preference set", or "reward-shaped frame-selection policy" are noise. VLM training is just LLM training with two added stages (vision-language alignment + visual instruction tuning), plus increasingly the same RL flavor on top — so getting the LLM training picture right pays off everywhere downstream.

## Prerequisites

Module 01 (decoder-only LMs, KV cache).

## The four-stage map

```
            ┌─────────────────┐
            │ 1. Pre-training │   massive web text + code, causal-LM loss
            │   ("base" LLM)  │   ──► broad knowledge, no instruction-following
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 2. SFT          │   curated (prompt, completion) pairs
            │   (instruction  │   ──► model follows instructions but
            │    tuning)      │       still hallucinates, can be unsafe
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 3. Preference   │   (prompt, chosen, rejected) triples
            │    alignment    │   PPO-style RLHF, or DPO, or GRPO…
            │                 │   ──► helpful, honest, and on-style
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ 4. (optional)   │   reasoning / tool-use / agentic training,
            │    Reasoning RL │   verifiable rewards. R1-style.
            └─────────────────┘
```

For VLMs, stages 1–4 stay the same conceptually but get two visual-specific *sub-stages*:

```
            (VLM additions)
            
            1.5  Vision-language alignment        train only the projector;
                  (frozen LLM + frozen encoder)   image-caption pairs
            
            2.5  Visual instruction tuning        train projector + LLM
                  (encoder still frozen, usually) image-conversation pairs
                                                  (GPT-4-generated, etc.)
            
            3.5  Multimodal preference alignment  RLHF / DPO with
                  (LLaVA-RLHF, RLHF-V)            image-grounded preferences
```

## Learning path

### 1. The big-picture video that ties it all together

If you watch one thing, watch this — chapters let you skip what you already know.

- **Primary:** [Andrej Karpathy — *Deep Dive into LLMs like ChatGPT* (YouTube, 3h31m, chaptered)](https://www.youtube.com/watch?v=7xTGNNLPyMI). For this module, watch the chapters on **pretraining**, **supervised fine-tuning**, and **reinforcement learning** — ~75 minutes if you skip what you know.
- **Go deeper:** [Karpathy — *Neural Networks: Zero To Hero*](https://karpathy.ai/zero-to-hero.html) — full course, only if you have time later.

### 2. Pre-training (Stage 1)

What it is: maximum-likelihood causal language modeling on a vast text corpus (trillions of tokens). Token-by-token next-word prediction. Produces a "base model" that can complete text but can't follow instructions. **You already know how to train this from Module 01** — the loss is just next-token cross-entropy.

What's special to know for the reading list:
- Modern open base models (Llama 3, Qwen 3, Mistral) are the LLMs *inside* every VLM you'll read about.
- Pre-training data quality and mix dominate base-model quality. This is why VLMs built on different LLM backbones behave noticeably differently even with the same visual ingredients.

There's nothing more to do at this stage — skip to SFT unless you specifically need to brush up.

### 3. Supervised fine-tuning / instruction tuning (Stage 2)

What it is: continue training the base model on a small high-quality dataset of (prompt, ideal-response) pairs. Same cross-entropy loss, just on curated data. Teaches the model to *follow instructions* (and adopt a tone, format, persona, etc.). This is the cheapest, fastest stage with the biggest behavioral change.

Key things to internalize:
- SFT alone gets you ~80% of "ChatGPT-feeling" behavior. RL on top is for the last mile (factuality, refusing harmful requests, style, safety).
- The data is the secret. The exact dataset choice — and the *order* of examples — matters more than the loss.

VLM-specific equivalent — **visual instruction tuning** (Stage 2.5):
- Pioneered by LLaVA: use GPT-4 to *generate* image-grounded conversations from captions. This synthetic data + a few public VQA sets is the entire SFT for many open VLMs.
- For Module 03, the LLaVA paper *is* the canonical SFT example for VLMs — read it as much for the data-construction trick as for the architecture.

- **Primary:** [Hugging Face — *Illustrating Reinforcement Learning from Human Feedback (RLHF)*](https://huggingface.co/blog/rlhf) — first half covers SFT cleanly with diagrams.
- **Go deeper:** Liu et al., *Visual Instruction Tuning (LLaVA)*, NeurIPS 2023 — [arXiv:2304.08485](https://arxiv.org/abs/2304.08485). Skim path: §3 (data construction) → §4 (training recipe). Re-encounter in Module 03.

### 4. RLHF with PPO (Stage 3, classical)

What it is, in one paragraph: collect preference data — humans (or a stronger model) compare two model outputs and pick the better one. Train a **reward model** to predict that preference. Then fine-tune the SFT model with **PPO** (a policy-gradient RL algorithm with a clipped surrogate objective) to maximize the learned reward, with a **KL penalty** pulling it back toward the SFT model so it doesn't go off the rails.

Why it's a pain:
- Three models in memory simultaneously (policy, reference, reward) — expensive.
- PPO is notoriously finicky: hyper-parameters, value heads, advantage estimation.
- The reward model can be gamed (reward hacking).

Why everyone did it anyway: it worked. InstructGPT (1.3B PPO'd) beat GPT-3 (175B). Established the field's intuition that "post-training is most of the value."

- **Primary:** [Hugging Face — *Illustrating RLHF*](https://huggingface.co/blog/rlhf) — covers the full PPO pipeline with diagrams.
- **Go deeper:** Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback (InstructGPT)*, 2022 — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). Skim path: §3 (method) — the three steps.

### 5. DPO and the "no reward model" alternative (Stage 3, modern default)

What it is: Rafailov et al. showed that the optimal RLHF policy can be expressed in **closed form** as a function of the SFT model and the preference data. That collapses the entire RLHF pipeline into a **single classification-style loss** on (prompt, chosen, rejected) triples — no separate reward model, no PPO, no value function. Just a fancy cross-entropy.

Practical implications:
- Stable, cheap, easy to implement (HuggingFace TRL has it in a few lines).
- Matches or beats PPO on most academic benchmarks.
- This is now the *default* preference-alignment recipe in open-source land.

For VLMs:
- **LLaVA-RLHF** (Fact-RLHF): classical PPO-style, factuality-augmented. The first open multimodal RLHF.
- **RLHF-V** (CVPR 2024): segment-level corrections + dense DPO. Targets hallucination specifically.
- Most newer open VLMs (Qwen2.5-VL, InternVL3, …) use DPO or its descendants on multimodal preference data.

- **Primary:** [HF blog — *Navigating the RLHF Landscape: PPO → DPO*](https://huggingface.co/blog/NormalUhr/rlhf-pipeline) — covers PPO *and* DPO side-by-side; lighter than the paper.
- **Go deeper:** Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023 — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290). Skim path: §4 (DPO derivation) is one page of math; §6 (experiments) shows it matches PPO.

### 6. GRPO + reasoning RL (Stage 4)

What it is: PPO normally needs a value-function critic to estimate advantages. GRPO ("Group Relative Policy Optimization") replaces the critic with a **simple group baseline**: sample K completions per prompt, use the average reward of the group as the baseline, normalize. Trains much more cheaply and at scale.

Why it matters now: DeepSeek used GRPO with **verifiable rewards** (does the model's answer pass the unit test? did it match the gold solution?) to train DeepSeek-R1 — the first major open "reasoning" model. The recipe ("RL with verifiable rewards on a strong base model") is the post-training fashion of 2025-2026.

Connections to your reading list:
- **VideoBrain** in Module 07 has an RL component — a *behavior-aware reward* shaping a frame-selection policy. That's the same family of ideas, applied to a VLM tool-use policy.
- Expect more reading-list candidates over time that use GRPO or its variants for VLM tasks.

- **Primary:** [Why GRPO is Important and How it Works (Oxen AI blog)](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/) — readable explainer with the loss formula.
- **Go deeper:** Shao et al., *DeepSeekMath (introduces GRPO)*, 2024 — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). Skim path: §4.1 (GRPO algorithm). Then DeepSeek-AI, *DeepSeek-R1*, 2025 — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) for the verifiable-reward application.

### 7. One-paragraph map of the rest

You don't need these for the reading list, but the names come up in 1:1s:
- **ORPO** (odds-ratio preference optimization): DPO-like but does SFT and preference learning in *one* loss. Useful when you have no SFT-only data, only preference data.
- **KTO** (Kahneman-Tversky Optimization): preference learning from *unpaired* binary "good / bad" labels (no need for chosen-vs-rejected pairs). Cheaper labeling.
- **REINFORCE++ / RLOO**: modern variants of plain REINFORCE that, like GRPO, drop the critic. Increasingly competitive.
- **RLAIF**: same loop as RLHF but the preferences come from a strong LLM, not humans. Cheap; common in 2024+.
- **Constitutional AI** (Anthropic): a recipe combining RLAIF with a written set of principles; how Claude is post-trained.

If you only remember one frame: **after pre-training, every modern model is a small training-data trick on top of an "SFT + (DPO or GRPO)" stack.** The rest is variations.

## Self-check

- **State the four stages in order and what each stage's training data looks like.** (1. Web text, causal LM. 2. (prompt, ideal-response) pairs, cross-entropy. 3. Preference triples, RLHF/DPO/GRPO. 4. Verifiable rewards on reasoning tasks, GRPO-style.)
- **Why is DPO simpler than PPO-RLHF?** (No separate reward model, no value function, no RL roll-outs. One supervised-style loss on preference triples.)
- **What's a "base model" vs. an "instruct model"?** (Base = stage 1 only, completes text but doesn't follow instructions. Instruct = at least stage 2; sometimes called "chat" or "it" suffixed models like `Qwen2.5-7B-Instruct`.)
- **For VLMs, what does stage 1.5 (vision-language alignment) train? What's frozen?** (Trains only the projector. LLM and vision encoder are frozen. Data: image-caption pairs.)
- **Mechanism question:** when a paper says "we use DPO on multimodal preference data" — what's actually in each row of that dataset? (A prompt + image, a *chosen* model response, and a *rejected* model response. The DPO loss pushes log-prob of chosen up, rejected down, relative to a reference model.)
- **Cross-module:** which Module 07 paper most directly uses reasoning-RL ideas, and how? (VideoBrain — its behavior-aware reward shaping is in the same family as GRPO with verifiable rewards.)

## My notes

(fill in as you go)

## References

- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback (InstructGPT)*, NeurIPS 2022 — arXiv:2203.02155.
- Schulman et al., *Proximal Policy Optimization Algorithms (PPO)*, 2017 — arXiv:1707.06347.
- Rafailov et al., *Direct Preference Optimization (DPO)*, NeurIPS 2023 — arXiv:2305.18290.
- Shao et al., *DeepSeekMath (introduces GRPO)*, 2024 — arXiv:2402.03300.
- DeepSeek-AI, *DeepSeek-R1*, 2025 — arXiv:2501.12948.
- Bai et al., *Constitutional AI*, 2022 — arXiv:2212.08073.
- Sun et al., *Aligning Large Multimodal Models with Factually Augmented RLHF (LLaVA-RLHF)*, 2023 — arXiv:2309.14525.
- Yu et al., *RLHF-V*, CVPR 2024 — arXiv:2312.10665.
