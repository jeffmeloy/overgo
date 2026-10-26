package sampling

import "math/bits"

// philoxUniform is Strata's philox_uniform (src/kernels/cuda/sampler.cu, Strata at 296b59d2, MIT): Philox 4x32-10
// over the counter and the seed, its first word's top 24 bits as a uniform in [0, 1). Counter-based, the draw is a
// function of (seed, position) rather than of the draws before it, so a verify window's rows, sampled after the
// window, draw what plain decoding draws at the same positions, whatever the drafts were.
func philoxUniform(seed, counter uint64) float64 {
	c0, c1 := uint32(counter), uint32(counter>>philoxWordBits)
	c2, c3 := uint32(seed), uint32(seed>>philoxWordBits)
	for round := range uint32(philoxRounds) {
		hi0, lo0 := bits.Mul32(philoxMultiplier0, c0)
		hi1, lo1 := bits.Mul32(philoxMultiplier1, c2)
		c0, c1, c2, c3 = hi1^c1^round, lo1, hi0^c3, lo0
	}
	return float64(float32(c0>>philoxDroppedBits) * philoxUniformScale)
}

// Philox 4x32-10's constants as Strata's kernel holds them: its multipliers, its ten rounds keyed by the round number
// in the first key word and zero in the second, a word's bits, and the 24 bits of mantissa a uniform keeps (the first
// word shifted down by 8, scaled by 2^-24, so the value never rounds to 1).
const (
	philoxMultiplier0  = 0x9E3779B9
	philoxMultiplier1  = 0xBB67AE85
	philoxRounds       = 10
	philoxWordBits     = 32
	philoxDroppedBits  = 8
	philoxUniformScale = 1 / 16777216.0
)

// drawAt keys the call's draws by the row it samples: a call given the sequence draws at the position of its last
// token (the row whose logits these are), as Strata's verify draws row t of a window at pos0 + t; a call without it
// draws from the sequential stream.
func (s *Sampler) drawAt(history []int) {
	s.positioned = len(history) > 0
	if s.positioned {
		s.drawCounter = uint64(len(history) - 1)
	}
}

// uniform is the call's draw in a counter space: Philox over the seed and the row's position when the call is
// positioned, the sequential stream otherwise.
func (s *Sampler) uniform(space uint64) float64 {
	if s.positioned {
		return philoxUniform(uint64(s.config.Seed), s.drawCounter|space)
	}
	return s.random.Float64()
}

// Strata's counter spaces for a row's draws (spec_prob.hpp): the plain pick at the row's position, a draft's
// acceptance and its residual pick in spaces of their own (kSpecCtrAccept, kSpecCtrResid), and the XTC filter's.
const (
	philoxCounterAccept   = uint64(1) << 62
	philoxCounterResidual = uint64(1) << 63
	philoxCounterXTC      = uint64(1) << 60
)
