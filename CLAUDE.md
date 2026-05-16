# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a personal study guide repository for the user (UC Berkeley, chanwutk@berkeley.edu) to build foundation knowledge in **vision-language models (VLMs)** before an internship starting **2026-06-01**. The internship topic is **efficient video inference via visual token reduction**.

The repo is not a software project. There is no application to build, run, or deploy. Deliverables are study materials: notes, summaries, paper write-ups, diagrams, and runnable example code that illustrates concepts.

## What "done" looks like for this repo

The user needs to walk into the internship comfortable with:
1. **VLM fundamentals** — how vision encoders (ViT, CLIP-style) connect to LLMs, projector/connector designs (Q-Former, MLP, cross-attention), training stages (alignment vs. instruction tuning), and major model families (LLaVA, BLIP-2, Qwen-VL, InternVL, etc.).
2. **Video VLMs** — how images extend to videos: frame sampling, temporal encoding, the explosion in visual tokens that motivates reduction.
3. **Visual token reduction** — the literature directly relevant to the internship: token pruning (FastV, VisionZip), merging (ToMe and video variants), pooling, KV-cache compression, query-based selection, and training-free vs. trained methods.
4. **Evaluation** — common video understanding benchmarks (Video-MME, MVBench, EgoSchema, LongVideoBench, NExT-QA) and the tradeoffs they expose (accuracy vs. tokens/sec vs. memory).

When generating study materials, bias content toward the internship topic. General VLM material should be framed as scaffolding for understanding video token reduction.

## Content conventions

When the user asks to add a topic, paper summary, or note:
- Prefer **editing existing files** over creating new ones. A single growing `notes/<topic>.md` beats many shallow files.
- Paper summaries should follow a consistent shape: **problem → key idea → method → results → why it matters for token reduction**. Be specific about numbers (tokens kept, FLOPs reduction, benchmark deltas) when the paper reports them.
- When explaining a method, prefer a small diagram (mermaid or ASCII) over prose-only descriptions. The user is a visual learner working toward a visual-token research problem.
- Cite papers as `Author et al., Venue Year` inline, with full references collected at the bottom of the file. Include arXiv IDs when available so the user can pull them up quickly.
- Code examples should be **minimal and runnable** (PyTorch unless otherwise specified). Aim for ~30 lines that illustrate the mechanism — not a full reproduction.

## Tone for study materials

The user is technically strong (Berkeley CS, ML background expected) but new to this specific subfield. Skip generic ML preamble (what attention is, what a transformer is). Spend the words on what's *specific* to VLMs and video token reduction: why a particular design choice was made, what failure mode it addresses, what the next paper changed and why.

## Things not to do

- Don't scaffold a build system, package manager, or test framework unless the user asks. This is a notes repo, not a codebase.
- Don't create placeholder files ("TODO: fill in later"). Either write the content or don't create the file.
- Don't pad summaries with restated abstracts. If a paper's contribution can be stated in three sentences, use three sentences.
