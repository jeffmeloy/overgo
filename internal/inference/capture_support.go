package inference

import "overgo/internal/model"

// SupportsHiddenStateCapture reports whether ExtractLayerInputs can run for the
// loaded architecture — the same guard ExtractLayerInputs applies: causal,
// non encoder-decoder, and not AltUp (Gemma3n). Lets the analysis surface gate
// the feature instead of failing a request. Read-only.
func (r *Runner) SupportsHiddenStateCapture() bool {
	if r == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return false
	}
	profile := r.profile()
	return !r.spec.NonCausalAttention &&
		profile.Family != model.ArchitectureFamilyEncoderDecoder &&
		!profile.Has(model.ArchitectureAltUp)
}
