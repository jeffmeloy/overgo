package optimizer

import (
	"math"

	"overgo/internal/tensor/dtype"
)

// BF16SGD is the at-scale training memory path: plain SGD on BF16 master weights
// (2 bytes/param, half an F32 master) with NO momentum state and NO F32 master
// copy. Gradients arrive F32 (accumulated in F32 upstream, where precision
// matters); the update expands each master weight to F32, applies -lr*grad, and
// rounds back to BF16. A naive round-to-nearest update stalls when lr*grad is
// small relative to the BF16 step at that magnitude (the low mantissa bits are
// discarded every step); Step therefore rounds STOCHASTICALLY, so an update
// smaller than one BF16 ULP still lands in expectation without carrying an F32
// residual. This is what lets a model that OOMs under F32-grad SGD (e4b ~48GB)
// train within a single-GPU memory bound.
type BF16SGD struct {
	master []uint16 // BF16 master weights, one per parameter
	rng    uint64   // xorshift state for stochastic rounding (deterministic per seed)
}

// NewBF16SGD seeds a stepper from F32 initial weights, storing them as BF16.
// seed makes stochastic rounding reproducible.
func NewBF16SGD(weights []float32, seed uint64) *BF16SGD {
	master := make([]uint16, len(weights))
	for i, w := range weights {
		master[i] = dtype.Float32ToBF16(w)
	}
	if seed == 0 {
		seed = 0x9e3779b97f4a7c15
	}
	return &BF16SGD{master: master, rng: seed}
}

// Weights expands the BF16 master back to F32 (for the forward pass / readout).
func (o *BF16SGD) Weights() []float32 {
	out := make([]float32, len(o.master))
	for i, m := range o.master {
		out[i] = dtype.BF16ToFloat32(m)
	}
	return out
}

// WeightsInto expands into dst (len must match) to avoid an allocation per step.
func (o *BF16SGD) WeightsInto(dst []float32) {
	for i, m := range o.master {
		dst[i] = dtype.BF16ToFloat32(m)
	}
}

// Master returns the BF16 master storage (len == parameter count); each element
// is 2 bytes, the residency win this path exists for.
func (o *BF16SGD) Master() []uint16 { return o.master }

// Step applies w <- roundStochastic(w - lr*grad) per parameter, in BF16.
func (o *BF16SGD) Step(grad []float32, lr float64) {
	for i := range o.master {
		updated := float64(dtype.BF16ToFloat32(o.master[i])) - lr*float64(grad[i])
		o.master[i] = o.roundBF16Stochastic(float32(updated))
	}
}

// roundBF16Stochastic rounds an F32 to BF16, choosing up or down with
// probability proportional to where the value falls between the two
// representable BF16 neighbours -- unbiased in expectation, so sub-ULP updates
// are preserved statistically without an F32 residual.
func (o *BF16SGD) roundBF16Stochastic(value float32) uint16 {
	bits := math.Float32bits(value)
	// NaN/Inf: fall back to round-to-nearest-even (dtype handles it).
	if bits&0x7f800000 == 0x7f800000 {
		return dtype.Float32ToBF16(value)
	}
	down := bits & 0xffff0000 // truncated toward zero in the exponent/mantissa grid
	rem := bits & 0x0000ffff  // the 16 low bits being discarded
	// probability of rounding up == rem/2^16.
	if uint32(o.next()&0xffff) < rem {
		down += 0x00010000 // step to the next BF16 (carries into mantissa/exponent correctly)
	}
	return uint16(down >> 16)
}

// next is a xorshift64* step -- deterministic, no global RNG, cheap per element.
func (o *BF16SGD) next() uint64 {
	x := o.rng
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	o.rng = x
	return x * 0x2545f4914f6cdd1d
}
