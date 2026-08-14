package hostmath

import "math"

// GatedDeltaNetForward is the host reference for the gated_delta_net_f32 kernel:
// the gated delta-rule linear-attention recurrence used by qwen3.5's
// linear_attention layers. Per (head, sequence) it carries a [size,size] state S
// (row-major), initialized from inputState, and for each token, independently per
// state row r:
//
//	S[r][c] *= exp(gate[c])              (per-column gate; scalar when gateWidth==1)
//	d       = (v[r] - S[r]·k) * beta     (delta rule error, gated by beta)
//	S[r][c] += d * k[c]
//	out[r]   = (S[r]·q) * (1/sqrt(size))
//
// Grouped q/k heads: query_head = repeatInterleave ? head/(heads/queryHeads) :
// head%queryHeads (same for key). Layouts match the kernel: query/key/value/gate
// are [sequence, token, heads, size] (query/key use their own head counts; gate
// width is gateWidth), beta is [sequence, token, heads]; output is
// [sequence, token, heads, size]; input/final state are [heads*sequences, size,
// size]. f64 accumulation makes this the differentiation target for the VJP.
func GatedDeltaNetForward(
	query, key, value, gate, beta, inputState []float32,
	size, queryHeads, keyHeads, heads, tokens, sequences, gateWidth int,
	repeatInterleave bool,
) (output, finalState []float32) {
	scale := 1.0 / math.Sqrt(float64(size))
	output = make([]float32, size*heads*tokens*sequences)
	finalState = make([]float32, heads*sequences*size*size)
	state := make([]float64, size*size)

	for sequence := 0; sequence < sequences; sequence++ {
		for head := 0; head < heads; head++ {
			index := sequence*heads + head
			src := inputState[index*size*size:]
			for i := 0; i < size*size; i++ {
				state[i] = float64(src[i])
			}
			queryHead := head % queryHeads
			keyHead := head % keyHeads
			if repeatInterleave {
				queryHead = head / (heads / queryHeads)
				keyHead = head / (heads / keyHeads)
			}
			for token := 0; token < tokens; token++ {
				valueBase := ((sequence*tokens+token)*heads + head) * size
				queryBase := ((sequence*tokens+token)*queryHeads + queryHead) * size
				keyBase := ((sequence*tokens+token)*keyHeads + keyHead) * size
				gateBase := ((sequence*tokens+token)*heads + head) * gateWidth
				betaValue := float64(beta[(sequence*tokens+token)*heads+head])
				for r := 0; r < size; r++ {
					row := state[r*size : r*size+size]
					// Gated decay of this state row.
					for c := 0; c < size; c++ {
						g := float64(gate[gateBase])
						if gateWidth != 1 {
							g = float64(gate[gateBase+c])
						}
						row[c] *= math.Exp(g)
					}
					// Delta rule: error against the value, gated by beta.
					var dot float64
					for c := 0; c < size; c++ {
						dot += row[c] * float64(key[keyBase+c])
					}
					delta := (float64(value[valueBase+r]) - dot) * betaValue
					for c := 0; c < size; c++ {
						row[c] += delta * float64(key[keyBase+c])
					}
					// Readout against the query.
					var readout float64
					for c := 0; c < size; c++ {
						readout += row[c] * float64(query[queryBase+c])
					}
					output[valueBase+r] = float32(readout * scale)
				}
			}
			dst := finalState[index*size*size:]
			for i := 0; i < size*size; i++ {
				dst[i] = float32(state[i])
			}
		}
	}
	return output, finalState
}
