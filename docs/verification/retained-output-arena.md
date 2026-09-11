# Retained-output arena repair

Retained CUDA execution allocated outputs separately while reserving their
unused arena slots. It now uses the existing planner's output exclusions.
Host-output execution retains its original plan. Kernels and numerical
operations are unchanged; no public API was added.

The current-source Gemma 12B FP8 guard stopped before its required 8192-token
rung. Its retained reference had peaked above the existing 90% device margin.
The repair completes that rung and passes the unchanged historical comparison.

| 8192-token FP8 guard | Owned peak bytes |
| --- | ---: |
| Retained reference | 48,654,126,784 |
| Three fresh repeats | 39,408,584,384 |
| Reduction | 9,245,542,400 |

The CPU counterexample reserved 8,192 duplicate arena bytes. Eight focused CPU
and CUDA checks pass, covering output slots, lifetime, aliases, replay and arena
accounting. Three clean-source FP8 repeats now pass every required guard rung
and the historical/first-repeat comparisons at producer
`95c0ac02654d9e9034451f9ab1a4d4b2abd0646b`. Each has the table's exact peak.
Judged decode: 30.1, 30.0, 30.0 tokens/s; measured wall: 97.438, 96.905,
96.867 seconds. The historical cohort remains in selection
`evidence:sha256:e970606101f9f1b185d15a0cda2f90d4f59ef7644c8909bc76f03a26e0233c69`.
`TestAcceptedFP8Guard` checks the current FP8 cohort in the shared
[guard catalog](guard-catalog.json).
This is a regression guard, not chat-quality, full-modality or MMLU acceptance.

Fresh E4B/Qwen controls and full text/media resource recovery also pass; see
[E4B validation](e4b-validation.md). Earlier evidence retains its original
source identity. The catalog-wide finding remains open until its complete
catalog check passes.

Probe, patch, raw output and failed-record lineage:
`evidence:sha256:e96160d7908a86b0c85211a656976f75d4bfba7a425b7bd73af2116df01235a6`.
