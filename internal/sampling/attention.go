package sampling

import (
	"fmt"

	"overgo/internal/checked"
)

// NormalizedAttention is NAG, normalized attention guidance (Chen et al., arXiv 2505.21179), written from its paper
// with its authors' code read for the one thing the paper leaves open, the norm's extent. It guides in attention space,
// so a model run at a guidance scale of one, with no unconditioned pass, still honours a negative prompt: in each
// cross-attention, the image tokens' output for the prompt, Z+, is extrapolated away from their output for the negative
// prompt, Z-, as Z~ = Z+ + Scale (Z+ - Z-) (Eq. 7); each token's L1 norm of Z~ over the prompt's, R (Eq. 8), is bounded
// at Tau, Z^ = min(R, Tau) / R Z~ (Eq. 9); and Z = Alpha Z^ + (1 - Alpha) Z+ (Eq. 10) goes on to the output projection.
// The norm is a token's, over all its heads' features together, as the authors' code flattens the heads before it.
// Their code also writes Eq. 7 as nag_scale Z+ - (nag_scale - 1) Z-, so a nag_scale is Scale + 1: their README's Wan
// example, nag_scale 9, is Scale 8, where the paper's Table 5 gives Wan2.1 Scale 4, Tau 2.5 and Alpha 0.25. Its zero
// value guides nothing.
type NormalizedAttention struct {
	Scale float64 `json:"scale"`
	Tau   float64 `json:"tau"`
	Alpha float64 `json:"alpha"`
}

// On is whether the guidance is NAG's.
func (n NormalizedAttention) On() bool { return n != NormalizedAttention{} }

// Validate refuses NAG no generation can run: a scale or bound that is no positive finite number, or a blend outside
// (0, 1], at zero of which the guided output is the prompt's alone.
func (n NormalizedAttention) Validate() error {
	if !n.On() {
		return nil
	}
	if !checked.PositiveFinite64(n.Scale) || !checked.PositiveFinite64(n.Tau) || !(n.Alpha > 0 && n.Alpha <= 1) {
		return fmt.Errorf("guidance: NAG of scale %g, bound %g and blend %g cannot run", n.Scale, n.Tau, n.Alpha)
	}
	return nil
}
