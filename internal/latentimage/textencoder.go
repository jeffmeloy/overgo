// Qwen3-VL text-encoder forward + selected-hidden-layer capture (host port).
// This is the encoder that PRODUCES the [textSeq, TextLayers, TextHidden]
// conditioning tensor consumed by Denoiser.textConditioning (the text_fusion +
// txt_in stream) -- the image-diffusion counterpart of latentvideo's UMT5
// textcond sibling. It closes the "NEEDS-HOOK" gap named in denoiser.go: the
// Qwen3-VL 36-layer encoder's SELECTED hidden states were previously not ported.
//
// LAYER SELECTION (config-cited, NOT literals). The 12 tapped layers are READ
// from model_index.json text_encoder_select_layers. Those values are
// hidden_states indices in the transformers/diffusers convention where
// hidden_states[0] is the token embedding and hidden_states[N] is the residual
// stream AFTER decoder layer N-1; so a selected value N is captured immediately
// after running decoder layer N-1 (valid range [1, HiddenLayers]). This mirrors
// adaptive extmodel encodeSelectedLayerTextCUDA exactly (capture[layer+1]).
// len(SelectLayers) == TransformerSpec.TextLayers is the fusion-boundary
// cross-check already asserted in crossCheckConfig.
//
// ARITHMETIC (Qwen3 decoder layer, mirrors adaptive runtime_bf16_causal_gqa):
//   - STANDARD RMSNorm: x/sqrt(mean(x^2)+eps) * weight -- NOT the DiT's
//     zero-centered (1+weight) form; eps = text_config.rms_norm_eps.
//   - GQA attention: Heads q / KVHeads kv, per-head q/k RMSNorm over head_dim,
//     rotate-half (Llama/NeoX) RoPE with theta = text_config rope_theta, a CAUSAL
//     mask, and scale 1/sqrt(head_dim). Qwen3-VL mrope collapses to standard rope
//     for a pure-text sequence (all three mrope position sections share the
//     sequential token index), so a single rope table over the token position is
//     exact for the text-conditioning path.
//   - SwiGLU MLP: down(silu(gate(x)) * up(x)); no biases anywhere.
//
// Host math accumulates in f64 (reference-oracle convention, matching denoiser/
// vae). Weights STREAM layer-by-layer (only the embedding rows for the actual
// tokens plus one layer's weights are resident at a time) so the ~3.7B-parameter
// text encoder runs CPU-only for structural/finiteness telemetry with bounded
// peak memory.
//
// ORACLE SCOPE: telemetry/structural only. Exact numeric parity against the CUDA
// serving path needs an adaptive-side dump hook (out of scope; see denoiser.go
// TestDenoiserExactG3IsHookGapped). This port asserts the tokenizer, the 12-layer
// SELECTION, encoder forward shapes + finiteness on the real checkpoint, and that
// the produced conditioning geometry matches the DiT text-stream input -- never
// bit-exact values. The Krea conditioner's chat prompt-template wrapping / pad-row
// layout / attention mask (adaptive's text.prompt_template.prefix/suffix +
// text.max_prompt_tokens) is reproduced by renderTextInput in
// textinput.go (dtc brick 1/3); this encoder still accepts a raw id slice, so a
// caller wiring the device text-conditioning path renders via renderTextInput
// first. The remaining exact-parity residual is the DEVICE bf16 encoder forward.

package latentimage

import (
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/tensor"
)

// textEncoderPrefix is the checkpoint namespace for the Qwen3-VL language model.
// The sibling visual.* tower is a SEPARATE capability (vision conditioning), not
// part of the text-conditioning encoder, and language_model.norm.weight feeds
// last_hidden_state only (never a tapped intermediate); neither is consumed here.
const textEncoderPrefix = "language_model."

// captureSlots maps a selected hidden_states value to its output slot, and
// validates the selection against the adaptive convention: strictly increasing
// values in [1, HiddenLayers] (index N captured after decoder layer N-1).
func captureSlots(e TextEncoderSpec) (map[int]int, error) {
	if !checked.StrictlyIncreasingOneBased(e.SelectLayers, e.HiddenLayers) {
		return nil, fmt.Errorf("textencoder: select layers %v not strictly increasing in [1,%d]", e.SelectLayers, e.HiddenLayers)
	}
	slots := make(map[int]int, len(e.SelectLayers))
	for i, v := range e.SelectLayers {
		slots[v] = i
	}
	return slots, nil
}

// attendedPositions gives each token of a sequence the position its rotary
// embedding turns by: the count of attended tokens up to and including it, less
// one, and not below zero. Padding takes no position. In the
// [prefix][prompt][pad][suffix] layout the suffix then follows the prompt,
// which is how the model was trained: its own pipeline builds positions as the
// running count of its attention mask, less one, and says that numbering every
// slot would give the suffix a shifted phase. A nil mask attends every token
// and gives the token indices.
func attendedPositions(seq int, mask []bool) []uint32 {
	positions := make([]uint32, seq)
	var attended uint32
	for token := range positions {
		if mask == nil || mask[token] {
			attended++
		}
		positions[token] = max(attended, tensor.SingletonExtent) - tensor.SingletonExtent
	}
	return positions
}

// ---- structural checkpoint verification (headers only) --------------------

// EncoderWitness: the structural oracle for the encoder port -- the selection
// indices, the tapped-layer capture map, and every consumed tensor's shape vs the
// real checkpoint.
type EncoderWitness struct {
	Checks       []Check
	SelectLayers []int
	CaptureAfter []int // decoder layer (0-based) each tap is captured after
	Intermediate int
	Tensors      int
}
