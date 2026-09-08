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
| Repaired diagnostic | 39,408,584,384 |
| Reduction | 9,245,542,400 |

The CPU counterexample reserved 8,192 duplicate arena bytes. Eight focused CPU
and CUDA checks pass, covering output slots, lifetime, aliases, replay and arena
accounting. The real-model run remains a precommit diagnostic: clean-source
controls, affected E4B resource evidence and the three-repeat FP8 cohort must
be readmitted after the producer lands. Earlier model evidence keeps its
original source identity.

Probe, patch, raw output and failed-record lineage:
`evidence:sha256:e96160d7908a86b0c85211a656976f75d4bfba7a425b7bd73af2116df01235a6`.
