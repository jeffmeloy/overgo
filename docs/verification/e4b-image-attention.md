# E4B image attention

The text model config declares the mask. E4B's `use_bidirectional_attention`
is explicitly null (causal); the 12B FP8 config declares `vision`. Both
projector runners consume the recipe-bound declaration. The vision encoder's
own attention is a separate operation.

`fixtures/e4b_vision/hidden_mask_golden.json` records a native BF16 SDPA
intervention: negate the last image-token embedding at language-model entry,
hold native per-layer inputs fixed, and capture the first image token after
layers 0 and 5. With the E4B declaration, maximum hidden-state differences are
0 and 0. Changing only the declaration to `vision` produces 0.03125 and
0.1171875. This control makes the probe sensitive to future-image influence.

The corresponding Go test, `TestE4BImageMaskCausality`, captures inputs to
layers 1 and 6. It requires zero future influence with the declared policy
and nonzero influence under the wrong-mask control. Go recomputes per-layer
inputs after perturbation, so the control magnitudes are not a numerical
parity requirement. This establishes causal influence parity for this probe,
not complete hidden-state parity, modality quality, or a universal mask proof.
Full-layer hidden-state changes under the wrong policy can inherit earlier
sliding-layer changes; they do not establish direct full-layer attention.

The capture pins config and image hashes and records all hidden vectors,
prompt IDs, intervention positions, Torch 2.12.0+cu126 and Transformers 5.14.1.
Native `transformers/models/gemma4/modeling_gemma4.py` SHA256:
`ccab8e2dd80b71e9ca34e2c87291e17c40a27c755006e554da2ebf70d6616916`.

The inspected llama.cpp checkout is
`42fc243060709331ff9b158a9ed2cbe37219ae83`, with a locally modified
`tools/mtmd/mtmd.cpp`, SHA256
`457ddd483987f4afc2086d869a62fcbe1cb24453270b3114c693a58b32635339`.
Its `mtmd_decode_use_non_causal` selects non-causal decoding for both GEMMA4V
and GEMMA4UV. That is a source comparison, not an executed llama.cpp hidden-state
comparison, and cannot be attributed to the clean commit alone. The native
capture supports E4B's declared causal policy despite this implementation
disagreement. Short-answer greedy matches are retained as protocol evidence,
not used to decide the mask.
