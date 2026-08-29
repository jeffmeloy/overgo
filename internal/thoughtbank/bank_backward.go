package thoughtbank

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// FastWeightBankReadGradients mirrors FastWeightBankRead's inputs.
type FastWeightBankReadGradients struct {
	DH          []float32 // [rows, d]
	DBank       []float32 // [slots, mem_dim]
	DFWA        []float32 // [na*r*d, mem_dim]
	DFWB        []float32 // [d*r, mem_dim]
	DFWO        []float32 // [d, d]
	DNormWeight []float32 // [d]
}

// FastWeightBankReadBackward differentiates FastWeightBankRead.
//
// The forward is a SEQUENTIAL scan: each slot's low-rank update accumulates in
// place into y, so slot s+1 reads the y that slot s produced. That makes the
// backward a reverse-mode scan over slots -- a residual chain, like an unrolled
// recurrence -- not an independent per-slot sum. The per-slot forward state
// (the y each slot READ, and its activations) is not recoverable from the final
// y, so it is saved during a forward recompute here rather than carried in
// (recompute from inputs, do not trust stale state).
//
// out = h + FWO . (y_final - y0), y0 = rmsNorm(h). The residual h reaches the
// output twice (directly, and through y0/delta), so DH accumulates both.
//
// The SwiGLU read clamps silu(zg)*zv to +-FastWeightClamp; the derivative is zero
// in the clamped region, mirroring the forward's BRANCH.
func FastWeightBankReadBackward(h, bank, dOut []float32, rows, slots int, w *FastWeightBankWeights) (*FastWeightBankReadGradients, error) {
	if err := w.Validate(); err != nil {
		return nil, err
	}
	d, r, mem := w.DModel, w.Rank, w.MemDim
	if len(h) != rows*d {
		return nil, fmt.Errorf("fast-weight bank backward: h has %d values, want %d", len(h), rows*d)
	}
	if len(bank) != slots*mem {
		return nil, fmt.Errorf("fast-weight bank backward: bank has %d values, want %d", len(bank), slots*mem)
	}
	if len(dOut) != rows*d {
		return nil, fmt.Errorf("fast-weight bank backward: dOut has %d values, want %d", len(dOut), rows*d)
	}
	na := activationProjectionCount(w.SwiGLU)
	ds := 1.0 / math.Sqrt(float64(d))
	rs := 1.0 / math.Sqrt(float64(r))
	eps := w.NormEps

	// ---- Forward recompute, saving per-slot state ----
	y0 := rmsNormNew(h, w.NormWeight, rows, d, eps)
	y := make([]float32, len(y0))
	copy(y, y0)

	yBefore := make([][]float32, slots) // y each slot READ
	aState := make([][]float64, slots)  // [na*r*d]
	bState := make([][]float64, slots)  // [d*r]
	zState := make([][]float64, slots)  // [rows*r] (post-activation, post-clamp)
	zgState := make([][]float64, slots) // SwiGLU pre-activation gate [rows*r]
	zvState := make([][]float64, slots) // SwiGLU value [rows*r]
	clamped := make([][]bool, slots)    // SwiGLU clamp mask [rows*r]

	for s := range slots {
		yBefore[s] = make([]float32, rows*d)
		copy(yBefore[s], y)
		slot := bank[s*mem : (s+1)*mem]
		a := make([]float64, na*r*d)
		for i := range a {
			a[i] = dot(w.FWA[i*mem:(i+1)*mem], slot)
		}
		bmat := make([]float64, d*r)
		for i := range bmat {
			bmat[i] = dot(w.FWB[i*mem:(i+1)*mem], slot)
		}
		z := make([]float64, rows*r)
		var zg, zv []float64
		var cl []bool
		if w.SwiGLU {
			zg = make([]float64, rows*r)
			zv = make([]float64, rows*r)
			cl = make([]bool, rows*r)
		}
		for t := range rows {
			row := y[t*d : (t+1)*d]
			for k := range r {
				if w.SwiGLU {
					var g, v float64
					for j := range d {
						yv := float64(row[j])
						g += a[k*d+j] * yv
						v += a[r*d+k*d+j] * yv
					}
					g *= ds
					v *= ds
					zg[t*r+k] = g
					zv[t*r+k] = v
					val := g / (1.0 + math.Exp(-g)) * v // silu(g)*v
					if val > FastWeightClamp {
						val = FastWeightClamp
						cl[t*r+k] = true
					} else if val < -FastWeightClamp {
						val = -FastWeightClamp
						cl[t*r+k] = true
					}
					z[t*r+k] = val
					continue
				}
				var acc float64
				for j := range d {
					acc += a[k*d+j] * float64(row[j])
				}
				z[t*r+k] = hostmath.GELUErf(acc * ds)
			}
			for j := range d {
				var upd float64
				for k := range r {
					upd += bmat[j*r+k] * z[t*r+k]
				}
				row[j] += float32(upd * rs)
			}
		}
		aState[s], bState[s], zState[s] = a, bmat, z
		zgState[s], zvState[s], clamped[s] = zg, zv, cl
	}

	// ---- Gradients ----
	g := &FastWeightBankReadGradients{
		DH:          make([]float32, rows*d),
		DBank:       make([]float32, slots*mem),
		DFWA:        make([]float32, len(w.FWA)),
		DFWB:        make([]float32, len(w.FWB)),
		DFWO:        make([]float32, len(w.FWO)),
		DNormWeight: make([]float32, len(w.NormWeight)),
	}

	// Reverse the delta projection: out = h + FWO.(y_final - y0).
	dy := make([]float64, rows*d)  // grad w.r.t. current y (starts at y_final)
	dy0 := make([]float64, rows*d) // grad w.r.t. y0 (delta term; scan start added later)
	for t := range rows {
		yf := y[t*d : (t+1)*d]
		y0r := y0[t*d : (t+1)*d]
		for j := range d {
			g.DH[t*d+j] += dOut[t*d+j] // direct residual h
			doj := float64(dOut[t*d+j])
			foRow := w.FWO[j*d : (j+1)*d]
			dfoRow := g.DFWO[j*d : (j+1)*d]
			for k := range d {
				delta := float64(yf[k]) - float64(y0r[k])
				dfoRow[k] += float32(doj * delta) // dFWO[j,k]
				dcontrib := doj * float64(foRow[k])
				dy[t*d+k] += dcontrib  // d y_final
				dy0[t*d+k] -= dcontrib // d y0 via -y0 in delta
			}
		}
	}

	// ---- Reverse scan over slots ----
	for s := slots - 1; s >= 0; s-- {
		a, bmat, z := aState[s], bState[s], zState[s]
		yPrev := yBefore[s]
		slot := bank[s*mem : (s+1)*mem]
		da := make([]float64, na*r*d)
		dbmat := make([]float64, d*r)
		dz := make([]float64, rows*r)

		// y_after[t,j] = yPrev[t,j] + rs * sum_k bmat[j,k]*z[t,k]
		for t := range rows {
			for j := range d {
				dyaj := dy[t*d+j]
				if dyaj == 0 {
					continue
				}
				for k := range r {
					dbmat[j*r+k] += rs * dyaj * z[t*r+k]
					dz[t*r+k] += rs * dyaj * bmat[j*r+k]
				}
			}
		}
		// Activation backward: z depends on yPrev via a.
		for t := range rows {
			yr := yPrev[t*d : (t+1)*d]
			for k := range r {
				dzk := dz[t*r+k]
				if w.SwiGLU {
					if clamped[s][t*r+k] {
						continue // clamp derivative is zero
					}
					gpre := zgState[s][t*r+k]
					vval := zvState[s][t*r+k]
					sig := 1.0 / (1.0 + math.Exp(-gpre))
					silu := gpre * sig
					siluPrime := sig * (1.0 + gpre*(1.0-sig))
					dg := dzk * siluPrime * vval // d/dzg
					dv := dzk * silu             // d/dzv
					// zg = ds * sum_j a_g[k,j]*yPrev[j]; zv = ds * sum_j a_v[k,j]*yPrev[j]
					for j := range d {
						da[k*d+j] += dg * ds * float64(yr[j])
						da[r*d+k*d+j] += dv * ds * float64(yr[j])
						dy[t*d+j] += (dg*a[k*d+j] + dv*a[r*d+k*d+j]) * ds
					}
					continue
				}
				// gelu: pre = ds * sum_j a[k,j]*yPrev[j]; z = gelu(pre)
				var pre float64
				for j := range d {
					pre += a[k*d+j] * float64(yr[j])
				}
				pre *= ds
				dpre := dzk * hostmath.GELUErfPrime(pre)
				for j := range d {
					da[k*d+j] += dpre * ds * float64(yr[j])
					dy[t*d+j] += dpre * a[k*d+j] * ds
				}
			}
		}
		// Hypernet: a = FWA.slot, bmat = FWB.slot.
		for i := range na * r * d {
			if da[i] == 0 {
				continue
			}
			base := i * mem
			for m := range mem {
				g.DFWA[base+m] += float32(da[i] * float64(slot[m]))
				g.DBank[s*mem+m] += float32(da[i] * float64(w.FWA[base+m]))
			}
		}
		for i := range d * r {
			if dbmat[i] == 0 {
				continue
			}
			base := i * mem
			for m := range mem {
				g.DFWB[base+m] += float32(dbmat[i] * float64(slot[m]))
				g.DBank[s*mem+m] += float32(dbmat[i] * float64(w.FWB[base+m]))
			}
		}
		// dy now holds grad w.r.t. yPrev = y before this slot; proceed to s-1.
	}

	// After the scan, dy is grad w.r.t. y0 (the scan's input). Add the delta term.
	for i := range dy0 {
		dy0[i] += dy[i]
	}
	dy0f := make([]float32, rows*d)
	for i := range dy0f {
		dy0f[i] = float32(dy0[i])
	}
	// y0 = rmsNorm(h, NormWeight); the hostmath owner writes dx and dscale.
	dHnorm := make([]float32, rows*d)
	hostmath.RMSNormBackward(dHnorm, g.DNormWeight, h, w.NormWeight, dy0f, rows, d, eps, false)
	for i := range dHnorm {
		g.DH[i] += dHnorm[i]
	}
	return g, nil
}
