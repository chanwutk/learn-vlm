# 08 — Evaluation

## Why this matters

Every paper in Module 07 reports numbers on the same handful of benchmarks. You need to know what each benchmark *actually measures* so that when a paper claims "+5% on LongVideoBench," you know what kind of failure mode that 5% is covering up. You also need the **efficiency** axis: tokens, latency, memory. A method that's "+5% but 3× slower" is a different paper from "+5% at the same cost."

Day 16 (May 31) — this is a skim module. Don't aim to memorize numbers; aim to know which benchmark is which and what the cost axes are.

## Prerequisites

Module 07 (you've seen these benchmark names in context).

## Learning path

### 1. The five core benchmarks

| Benchmark | Length range | What it tests | Key feature for token-reduction work |
|---|---|---|---|
| **Video-MME** | 11 s – 60 min | General video QA across 6 domains | The "go-to" headline number on most 2025 papers; spans short / medium / long subsets. |
| **MVBench** | short clips | Temporal reasoning, action recognition | Short-form temporal capabilities; reductions that drop temporal detail show up here first. |
| **EgoSchema** | up to 180 s | Long-form ego-centric reasoning | First-person, complex social dynamics; "long" by 2023 standards. |
| **LongVideoBench** | avg 8 min, up to ~1 hr | Long-video understanding | Pushes the limits of frame budgets — where adaptive sampling shines. |
| **NExT-QA** | short clips | Causal / temporal / descriptive QA | Old (CVPR'21) but still common; tests whether reduction preserves *causal* reasoning, not just recognition. |

A useful rule of thumb when reading a results table:
- **Short benchmarks (MVBench, NExT-QA)**: token reduction is "free" — there are few tokens to begin with, so almost any method preserves accuracy.
- **Medium (Video-MME, EgoSchema)**: this is where differences show up. Methods that ablated only at low reduction ratios start to fall apart at higher ones.
- **Long (LongVideoBench)**: adaptive frame selection > spatial reduction. If a paper's only on the short benchmarks, treat its "wins" with caution.

- **Primary:** [Video-MME project page](https://video-mme.github.io/home_page.html) — has the benchmark composition table; ~5-min skim.
- **Go deeper:** [VideoLLM Benchmarks and Evaluation: A Survey](https://arxiv.org/html/2505.03829v1) — comprehensive map of the space.

### 2. The efficiency axes

Token-reduction papers report along three axes (sometimes four). When reading a table, make sure you check all of them:

| Axis | What it captures | How papers report it |
|---|---|---|
| **Accuracy** | Did the answer stay correct? | Per-benchmark score (often as a delta vs. no-reduction baseline). |
| **Token count** | How many tokens entered the LLM? | Absolute number, or "% of original." Headline number for almost every paper. |
| **TTFT (time-to-first-token)** | Prefill speed | Wall-clock seconds; only meaningful if measured with the same hardware. FastVLM and EVS push hard on this axis. |
| **Throughput / decode latency** | Per-output-token speed | Tokens/sec during generation. Decode-side methods (DyCoke stage 2, H2O) show up here. |
| **Memory** | Peak GPU memory used | Often reported as a separate column; KV-cache-side methods win here. |

A practical heuristic: **TTFT improvements are easy to game** (you can change batch size, hardware, framework). Token-count improvements are reproducible. When in doubt, trust the token-count number and look at the absolute hardware setup before believing a TTFT number.

### 3. The two patterns of "winning"

Pattern A: **Pareto improvement.** Same accuracy, fewer tokens / faster / less memory. Easy to celebrate; rare to find without an asterisk.

Pattern B: **Trade-off slider.** A knob (e.g., keep-ratio) that lets you smoothly choose accuracy vs. cost. Most reading-list papers report a *curve* across keep-ratios. The interesting question is the *shape* of the curve: where does it start dropping off?

When you discuss a paper, look at the **knees** of the curve — the point where accuracy starts collapsing. The interesting research questions usually live near the knee, not at the safe Pareto-improvement end.

- **Primary:** [Breaking Down Video LLM Benchmarks (recent survey)](https://arxiv.org/pdf/2505.14321) — sections on what each benchmark measures.
- **Go deeper:** Read the results sections of any two Module-07 papers side by side. Are they reporting at the same keep-ratio? On the same backbone? If not, the comparison is partially apples-to-oranges.

## Self-check

- **For each of the five benchmarks, state one sentence on what it tests and a representative video length.**
- **Why is a +5% on LongVideoBench more impressive than +5% on MVBench?** (LongVideoBench is harder; methods that preserve long-range reasoning are rarer than those that preserve short-clip recognition.)
- **If a paper reports "2× speedup on TTFT" but doesn't say what GPU, batch size, or framework — what's the right amount of skepticism?** (High. TTFT is the most-gameable axis. Prefer token-count or matched-hardware comparisons.)
- **You're given two methods at 30% token retention: method A keeps Video-MME accuracy within 1 point, method B keeps it within 0.5 point but is 2× slower. Which do you ship — and what's the missing piece of information?** (Depends on the deployment. The missing info: what's the use case's accuracy threshold and latency budget? Without that you can't choose.)
- **Mechanism question:** if every reading-list paper benchmarks on Video-MME, what's the risk? (Overfitting the field to one benchmark. The reading list also asks you to look at MVBench / EgoSchema / LongVideoBench / NExT-QA for exactly this reason — diversification of evaluation pressure.)

## Final review (May 31)

Last-day checklist before Day 1:
- [ ] `STUDY_GUIDE.md` progress tracker — every module checked off (or honestly partially-checked, with a note in `MANAGER_QUESTIONS.md`).
- [ ] All 11 papers from `SUGGESTED_READINGS.md` checked off.
- [ ] `MANAGER_QUESTIONS.md` has at least 5 questions and 2 half-baked ideas.
- [ ] You can sketch the canonical VLM pipeline diagram (Module 00) on a whiteboard from memory.
- [ ] You can sketch FastV and ToMe on a whiteboard from memory.
- [ ] Skim the bottoms of each module — your "My notes" sections — and pull the 3 best insights into a "lessons that surprised me" section of `MANAGER_QUESTIONS.md`.

## My notes

(fill in as you skim each benchmark)

## References

- Fu et al., *Video-MME*, CVPR 2025 — arXiv:2405.21075.
- Li et al., *MVBench*, CVPR 2024 — arXiv:2311.17005.
- Mangalam et al., *EgoSchema*, NeurIPS 2023 — arXiv:2308.09126.
- Wu et al., *LongVideoBench*, NeurIPS 2024 — arXiv:2407.15754.
- Xiao et al., *NExT-QA*, CVPR 2021 — arXiv:2105.08276.
- *VideoLLM Benchmarks and Evaluation: A Survey* — arXiv:2505.03829.
