# Colibri2 plan adoption

The worktree starts from master 0aa9853d0454caf6391538232da5344876c07135. The starting task inventory is
docs/plan.json from codex/overgo_colibri at e18cf921836bd4800796d3c14e41166dd6efb9ea,
with plan blob b0185e2adb5f889054945c59eb74ed2467043466. This adoption retains its
open capability tasks, updates their dependencies, and preserves master rows as
foreign owned until the lane projection has gated authority. Foreign rows grant
no completion credit and are not Colibri2 dispatch work. The new worktree's
independent OvergoDB store is seeded by a verified backup of master's ancestor
evidence; later Colibri2 results stay local to this worktree.

## Decisions from current master

Master integrated the Colibri EOS, Jinja template index, reasoning-channel and
evaluation-metadata work. Its merge deliberately excluded NFC and pre-tokenizer
activation after RxBrain VQA broke. The rollback also removed the previous
Unicode corpus and token oracle fixtures from the active tree; they remain in
Git history for reviewed restoration, not as current acceptance. The active
RxBrain tokenizer SHA256 is
AE5CA95EEB8E9A8774513A996E4E820A05C27DB32966AA8E071C10828402CB73.
It declares an empty normalization Sequence and three ordered isolated Split
patterns (numbers, CJK, a Unicode/mark/punctuation expression) before ByteLevel.
The third expression includes lookahead semantics. Its previous successful
production VQA path supplies the model-level regression test. General regex
support requires separate semantic parity; no silent fallback is accepted.

Master also replaced layered media reconciliation with
docs/media_reviewed_deltas.json. The old Colibri media-reconciliation item
was removed; remaining media work must use that registry. GUI cleanup is
integrated; runtime and expert views remain only for new measured observations
in the existing workbench. Prior-lane dependency references to pruned steps
were removed because this new worktree has no authority to credit those steps.

## Execution order

1. Restore shared NFC with Unicode 16 and an identity empty Sequence.
2. Extend declared ordered splits for actual RxBrain parity, then activate the
   compiler only with independent IDs and the real VQA golden passing. Check
   Krea, Fractale, Granite and the retained llama.cpp tokenizer corpora too.
3. Re-freeze complete encoding benchmarks on the resulting implementation,
   then fix integer-rank BPE merging through the common native owner.
4. Complete remaining Kimi/normalizer/model fixtures and compare measured
   serving, cache, placement and quality improvements. Obtain a real MoE
   numerical oracle before expert offload or grouping.

The plan's verify commands name future implementation obligations. No missing
named test or unavailable artifact counts as a pass.
