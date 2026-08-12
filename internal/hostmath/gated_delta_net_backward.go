package hostmath

import "math"

// GatedDeltaNetBackward is the VJP of GatedDeltaNetForward: given dOutput (the
// cotangent of the [seq,token,heads,size] output), it returns the gradients of
// every forward input. Derived by BPTT through the per-(head,sequence,row)
// recurrence; rows are independent so each carries its own [size] state. Per
// token, reversed:
//
//	dSnew  = dS_carry + dOut_t * scale * q_t          (dS_carry from token t+1)
//	dQ_t  += dOut_t * scale * s_t                     (s_t = state after update)
//	dDelta = dSnew · k_t ; dK_t += delta_t * dSnew
//	dV_t[r] += dDelta*beta_t ; dBeta_t += dDelta*(v_t[r]-dot_t)
//	dSp    = dSnew - beta_t*(dSnew·k_t)*k_t           (through delta + dot)
//	dK_t  += -beta_t*(dSnew·k_t) * s'_t
//	dGate_t[c] += dSp[c]*s'_t[c] ; dS_carry[c] = dSp[c]*exp(gate_t[c])
//
// grouped q/k heads accumulate (GQA). Saves s'_t and s_t per token; dot_t/delta_t
// are recomputed. f64 accumulation matches the forward.
func GatedDeltaNetBackward(
	query, key, value, gate, beta, inputState, dOutput []float32,
	size, queryHeads, keyHeads, heads, tokens, sequences, gateWidth int,
	repeatInterleave bool,
) (dQuery, dKey, dValue, dGate, dBeta, dInputState []float32) {
	scale := 1.0 / math.Sqrt(float64(size))
	dQuery = make([]float32, len(query))
	dKey = make([]float32, len(key))
	dValue = make([]float32, len(value))
	dGate = make([]float32, len(gate))
	dBeta = make([]float32, len(beta))
	dInputState = make([]float32, len(inputState))

	sprime := make([][]float64, tokens) // s'_t per token (this row)
	safter := make([][]float64, tokens) // s_t per token (this row)
	for t := range sprime {
		sprime[t] = make([]float64, size)
		safter[t] = make([]float64, size)
	}
	gexp := make([]float64, size)
	s := make([]float64, size)
	ds := make([]float64, size)
	dsnew := make([]float64, size)
	dsp := make([]float64, size)

	gateAt := func(base, c int) float64 {
		if gateWidth == 1 {
			return float64(gate[base])
		}
		return float64(gate[base+c])
	}

	for sequence := 0; sequence < sequences; sequence++ {
		for head := 0; head < heads; head++ {
			index := sequence*heads + head
			queryHead := head % queryHeads
			keyHead := head % keyHeads
			if repeatInterleave {
				queryHead = head / (heads / queryHeads)
				keyHead = head / (heads / keyHeads)
			}
			base := func(t, hh, hc int) int { return ((sequence*tokens+t)*hh + hc) * size }
			gbase := func(t int) int { return ((sequence*tokens + t) * heads * gateWidth) + head*gateWidth }
			bbase := func(t int) int { return (sequence*tokens+t)*heads + head }

			for r := 0; r < size; r++ {
				// Forward recompute for this row, saving s' and s per token.
				src := inputState[index*size*size+r*size:]
				for c := 0; c < size; c++ {
					s[c] = float64(src[c])
				}
				for t := 0; t < tokens; t++ {
					gb, kb, vb := gbase(t), base(t, keyHeads, keyHead), base(t, heads, head)
					betaValue := float64(beta[bbase(t)])
					var dot float64
					for c := 0; c < size; c++ {
						sp := s[c] * math.Exp(gateAt(gb, c))
						sprime[t][c] = sp
						dot += sp * float64(key[kb+c])
					}
					delta := (float64(value[vb+r]) - dot) * betaValue
					for c := 0; c < size; c++ {
						s[c] = sprime[t][c] + delta*float64(key[kb+c])
						safter[t][c] = s[c]
					}
				}
				// Backward reverse over tokens.
				for c := 0; c < size; c++ {
					ds[c] = 0
				}
				for t := tokens - 1; t >= 0; t-- {
					qb := base(t, queryHeads, queryHead)
					kb, vb := base(t, keyHeads, keyHead), base(t, heads, head)
					gb, bb := gbase(t), bbase(t)
					betaValue := float64(beta[bb])
					goVal := float64(dOutput[vb+r]) * scale
					// dSnew = ds + goVal*q ; dQ += goVal*s_t
					var dDelta float64
					for c := 0; c < size; c++ {
						dsnew[c] = ds[c] + goVal*float64(query[qb+c])
						dQuery[qb+c] += float32(goVal * safter[t][c])
						dDelta += dsnew[c] * float64(key[kb+c])
					}
					// recompute dot_t from s'_t for dBeta
					var dot float64
					for c := 0; c < size; c++ {
						dot += sprime[t][c] * float64(key[kb+c])
					}
					delta := (float64(value[vb+r]) - dot) * betaValue
					dValue[vb+r] += float32(dDelta * betaValue)
					dBeta[bb] += float32(dDelta * (float64(value[vb+r]) - dot))
					dDot := -dDelta * betaValue
					// dSp = dSnew + dDot*k ; dK += delta*dSnew + dDot*s'
					for c := 0; c < size; c++ {
						dsp[c] = dsnew[c] + dDot*float64(key[kb+c])
						dKey[kb+c] += float32(delta*dsnew[c] + dDot*sprime[t][c])
					}
					// s'_t = s_{t-1} ⊙ exp(gate): dGate += dSp*s' ; ds = dSp*exp(gate)
					for c := 0; c < size; c++ {
						gexp[c] = math.Exp(gateAt(gb, c))
						if gateWidth == 1 {
							dGate[gb] += float32(dsp[c] * sprime[t][c])
						} else {
							dGate[gb+c] += float32(dsp[c] * sprime[t][c])
						}
						ds[c] = dsp[c] * gexp[c]
					}
				}
				for c := 0; c < size; c++ {
					dInputState[index*size*size+r*size+c] = float32(ds[c])
				}
			}
		}
	}
	return dQuery, dKey, dValue, dGate, dBeta, dInputState
}
