package hostmath

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestGDNBackwardFD FD-grad-checks GatedDeltaNetBackward against central
// differences of L = sum(dOutput ⊙ GatedDeltaNetForward(...).output), for both
// scalar (gateWidth 1) and per-column gates.
func TestGDNBackwardFD(t *testing.T) {
	const size, qHeads, kHeads, heads, tokens, seqs = 4, 2, 2, 2, 3, 1
	for _, gateWidth := range []int{1, size} {
		t.Run(fmt.Sprintf("gateWidth=%d", gateWidth), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(7 + gateWidth)))
			rs := func(n int, scaleValue float64) []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = float32(rng.NormFloat64() * scaleValue)
				}
				return s
			}
			query := rs(size*qHeads*tokens*seqs, 0.3)
			key := rs(size*kHeads*tokens*seqs, 0.3)
			value := rs(size*heads*tokens*seqs, 0.3)
			gate := rs(gateWidth*heads*tokens*seqs, 0.1) // small: exp amplifies
			beta := rs(heads*tokens*seqs, 0.3)
			inputState := rs(heads*seqs*size*size, 0.3)
			dOutput := rs(size*heads*tokens*seqs, 1.0)

			dQ, dK, dV, dG, dB, dS := GatedDeltaNetBackward(
				query, key, value, gate, beta, inputState, dOutput,
				size, qHeads, kHeads, heads, tokens, seqs, gateWidth, false)

			loss := func() float64 {
				out, _ := GatedDeltaNetForward(query, key, value, gate, beta, inputState,
					size, qHeads, kHeads, heads, tokens, seqs, gateWidth, false)
				var l float64
				for i := range out {
					l += float64(dOutput[i]) * float64(out[i])
				}
				return l
			}
			const eps = 1e-3
			check := func(name string, arr, grad []float32) {
				var maxd float64
				for i := range arr {
					o := arr[i]
					arr[i] = o + float32(eps)
					lp := loss()
					arr[i] = o - float32(eps)
					lm := loss()
					arr[i] = o
					fd := (lp - lm) / (2 * eps)
					if d := math.Abs(fd - float64(grad[i])); d > maxd {
						maxd = d
					}
				}
				t.Logf("%s max|analytic-fd| %.3e", name, maxd)
				if maxd > 1e-2 {
					t.Fatalf("%s grad mismatch %.3e > 1e-2", name, maxd)
				}
			}
			check("dQuery", query, dQ)
			check("dKey", key, dK)
			check("dValue", value, dV)
			check("dGate", gate, dG)
			check("dBeta", beta, dB)
			check("dInputState", inputState, dS)
		})
	}
}
