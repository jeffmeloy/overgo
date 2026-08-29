// Package thoughtbank is the host (CPU, f32-storage, f64-accumulation)
// reference for the fast-weight thought-bank language model (adaptive's
// ThoughtBankLM / Fractale arch). Neutral home: nothing here names a model
// family — every dimension is supplied by the caller, derived from the
// checkpoint. Ported from adaptive's verified go/extmodel lane at b8fef3cc2 and
// re-expressed through overgo's owners (internal/hostmath for the shared math,
// internal/pytorchzip for the .pt read).
package thoughtbank

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// Fast-weight thought-bank read: memory as WEIGHTS, not as data.
//
// Ordinary memory-augmented attention reads a bank by attending over it, so the
// bank contributes values. Here each bank slot is expanded by a learned
// hypernet into the weights of a low-rank MLP layer, and the token stream is
// passed THROUGH the resulting stack -- slot i becomes layer i:
//
//	z = act(A_i y / sqrt(d));  y <- y + B_i z / sqrt(r)
//
// applied sequentially over the M slots, so the bank composes an M-layer
// network whose parameters the model itself wrote. The activation between slots
// is what makes that composition non-linear; without it, stacking slots
// collapses to a single low-rank linear map and the bank's depth buys nothing.
//
// The read is a DELTA against the normalised input (h + fw_o(y - y0)), so a
// bank carrying nothing is close to the identity rather than a perturbation.

// FastWeightClamp bounds the gated product in the SwiGLU read. It is 8, not the
// 10 used by the MoE feed-forward: the two clamps are separate constants in the
// reference and share no owner, so they are kept separate here rather than
// unified into one that would silently change whichever site moved.
const FastWeightClamp = 8.0

// FastWeightBankWeights is one block's hypernet, in reference layout.
type FastWeightBankWeights struct {
	DModel int
	MemDim int
	Rank   int
	// SwiGLU selects the gated read: fw_A emits TWO maps per slot (a gate and a
	// value) instead of one, and the product is clamped. The shipped
	// Fractale-350M checkpoint sets this true.
	SwiGLU bool

	FWA        []float32 // [na*r*d, mem_dim] with na = 2 when SwiGLU else 1
	FWB        []float32 // [d*r, mem_dim]
	FWO        []float32 // [d, d] output projection of the delta
	NormWeight []float32 // [d]
	NormEps    float64
}

// Validate reports a bundle whose shapes disagree with the declared dimensions.
func (weights *FastWeightBankWeights) Validate() error {
	na := activationProjectionCount(weights.SwiGLU)
	if weights.DModel < 1 || weights.MemDim < 1 || weights.Rank < 1 {
		return fmt.Errorf("fast-weight bank: d=%d mem_dim=%d rank=%d must be positive", weights.DModel, weights.MemDim, weights.Rank)
	}
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"fw_A", len(weights.FWA), na * weights.Rank * weights.DModel * weights.MemDim},
		{"fw_B", len(weights.FWB), weights.DModel * weights.Rank * weights.MemDim},
		{"fw_o", len(weights.FWO), weights.DModel * weights.DModel},
		{"norm_fw", len(weights.NormWeight), weights.DModel},
	} {
		if c.got != c.want {
			return fmt.Errorf("fast-weight bank: %s has %d values, want %d", c.name, c.got, c.want)
		}
	}
	return nil
}

// FastWeightBankRead applies the bank to a token stream.
//
// h holds rows x d values; bank holds slots x mem_dim values (one conversation,
// so the bank is not batched here -- callers with a batch apply this per row
// group, matching how the reference broadcasts a per-sequence bank).
func FastWeightBankRead(h, bank []float32, rows, slots int, w *FastWeightBankWeights) ([]float32, error) {
	if err := w.Validate(); err != nil {
		return nil, err
	}
	d, r := w.DModel, w.Rank
	if len(h) != rows*d {
		return nil, fmt.Errorf("fast-weight bank: input has %d values, want %d", len(h), rows*d)
	}
	if len(bank) != slots*w.MemDim {
		return nil, fmt.Errorf("fast-weight bank: bank has %d values, want %d", len(bank), slots*w.MemDim)
	}
	na := activationProjectionCount(w.SwiGLU)

	y0 := rmsNormNew(h, w.NormWeight, rows, d, w.NormEps)
	y := make([]float32, len(y0))
	copy(y, y0)

	// Per-slot hypernet expansion, reused across rows: the slot's weights do not
	// depend on the token, so expanding once per slot rather than per (slot,
	// token) is the difference between M and M*rows expansions.
	a := make([]float64, na*r*d)
	bmat := make([]float64, d*r)
	z := make([]float64, r)
	for s := range slots {
		slot := bank[s*w.MemDim : (s+1)*w.MemDim]
		for i := range na * r * d {
			a[i] = dot(w.FWA[i*w.MemDim:(i+1)*w.MemDim], slot)
		}
		for i := range d * r {
			bmat[i] = dot(w.FWB[i*w.MemDim:(i+1)*w.MemDim], slot)
		}

		ds := 1.0 / math.Sqrt(float64(d))
		rs := 1.0 / math.Sqrt(float64(r))
		for t := range rows {
			row := y[t*d : (t+1)*d]
			for k := range r {
				if w.SwiGLU {
					var zg, zv float64
					for j := range d {
						yv := float64(row[j])
						zg += a[k*d+j] * yv
						zv += a[r*d+k*d+j] * yv
					}
					zg *= ds
					zv *= ds
					v := zg / (1.0 + math.Exp(-zg)) * zv // silu(zg) * zv
					if v > FastWeightClamp {
						v = FastWeightClamp
					} else if v < -FastWeightClamp {
						v = -FastWeightClamp
					}
					z[k] = v
					continue
				}
				var acc float64
				for j := range d {
					acc += a[k*d+j] * float64(row[j])
				}
				z[k] = hostmath.GELUErf(acc * ds)
			}
			for j := range d {
				var upd float64
				for k := range r {
					upd += bmat[j*r+k] * z[k]
				}
				row[j] += float32(upd * rs)
			}
		}
	}

	// Delta form: only the CHANGE the bank produced is projected back.
	out := make([]float32, rows*d)
	delta := make([]float64, d)
	for t := range rows {
		for j := range d {
			delta[j] = float64(y[t*d+j]) - float64(y0[t*d+j])
		}
		for j := range d {
			var acc float64
			for k := range d {
				acc += float64(w.FWO[j*d+k]) * delta[k]
			}
			out[t*d+j] = h[t*d+j] + float32(acc)
		}
	}
	return out, nil
}

func fastWeightBankReadVector(h, bank []float32, slots int, weights *FastWeightBankWeights) ([]float32, error) {
	return FastWeightBankRead(h, bank, len(h)/weights.DModel, slots, weights)
}

func activationProjectionCount(swiGLU bool) int {
	if swiGLU {
		return 2
	}
	return 1
}

// rmsNormNew: allocating RMSNorm through the hostmath owner.
func rmsNormNew(x, weight []float32, rows, d int, eps float64) []float32 {
	out := make([]float32, rows*d)
	hostmath.RMSNormInto(out, x, weight, rows, d, eps)
	return out
}

func rmsNormVector(x, weight []float32, d int, eps float64) []float32 {
	return rmsNormNew(x, weight, len(x)/d, d, eps)
}

// dot is the f64-accumulated inner product of two f32 vectors.
func dot(w, x []float32) float64 {
	var acc float64
	for i, v := range w {
		acc += float64(v) * float64(x[i])
	}
	return acc
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

func tanh32(x float32) float32 { return float32(math.Tanh(float64(x))) }
